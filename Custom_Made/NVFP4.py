"""
NVFP4 (NVIDIA FP4) fake-quantization benchmark for a Hugging Face causal LM.

NVFP4 is NVIDIA's FP4 microscaling format. It differs from MXFP4 in two ways:

    1. Smaller blocks: 16 elements share a scale, not 32.
    2. A real (not power-of-two) block scale: the per-block scale is itself a genuine
       FP8 E4M3 number (max 448), so it can take any of FP8's ~240 positive values
       instead of only a power of two. This is why NVFP4 is usually more accurate
       than MXFP4 at almost the same cost.

Because a lone FP8 scale can't cover a whole tensor's dynamic range on its own, NVFP4
adds one extra FP32 "tensor scale" per weight matrix, shared by every block in it:

    dequantized_weight = fp4_value * (fp8_block_scale * fp32_tensor_scale)

Effective cost = 4 bits (element) + 8 bits / 16 (block scale) + a few bytes/tensor
(the tensor scale, amortized over the whole matrix) = 4.5 bits/weight.

Pipeline:
    FP32 weights
        -> one FP32 tensor_scale per Linear layer, from that layer's global |max|
        -> per-block (16) ideal scale, quantized into FP8 E4M3 relative to tensor_scale
        -> weights rounded to the FP4 E2M1 grid using the resulting two-level scale
        -> stored as packed 4-bit codes (uint8) + FP8 block scales + one FP32 tensor scale
        -> on every forward: unpack FP4 via a lookup table, rebuild each block's scale,
           multiply -> FP32 weight
        -> FP32 F.linear

This is "fake" quantization: the Ryzen 7 8845HS CPU and the Radeon 780M have no native
NVFP4 math (it is a Blackwell-generation NVIDIA GPU format), so this measures the
FORMAT (size, reconstruction error), not real NVFP4 speed.

The benchmark harness (timing, memory, perplexity) is IDENTICAL to the other scripts in
this series (FP32 / INT32 / BF16 / FP8 E4M3 / FP4 E2M1 / MXFP4), so the JSON results can
be compared directly. Only the quantization section differs.
"""

import gc
import json
import math
import platform
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

if not hasattr(torch, "float8_e4m3fn"):
    raise RuntimeError("This PyTorch build has no torch.float8_e4m3fn (needs PyTorch >= 2.1).")

# ============================================================
# Configuration  (keep identical to benchmark_fp32.py)
# ============================================================

MODEL_NAME = "EleutherAI/pythia-160m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # ROCm also shows up as "cuda"
DTYPE = torch.float32
ATTN_IMPL = "sdpa"
SEED = 0

# NVFP4 block size is fixed by the spec at 16 -- half of MXFP4's 32.
NV_BLOCK_SIZE = 16

BATCH_SIZE = 1
NUM_WARMUP = 10
NUM_ITERATIONS = 50

PREFILL_LENGTHS = [32, 128, 512]
DECODE_PROMPT_LEN = 32
DECODE_NEW_TOKENS = 64
DECODE_RUNS = 5

MAX_EVAL_LEN = 1024
USE_WIKITEXT = False               # True = WikiText-2 test set (needs `pip install datasets` + internet)

CPU_THREADS = 8                    # Ryzen 7 8845HS: 8 physical cores
RESULTS_FILE = "benchmark_nvfp4.json"


# ============================================================
# Reproducibility / true-FP32 settings
# ============================================================

torch.manual_seed(SEED)
torch.set_float32_matmul_precision("highest")
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
torch.set_num_threads(CPU_THREADS)


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


# ============================================================
# Helpers (identical to the FP32 baseline)
# ============================================================

MB = 1024 ** 2


def summarize(times_ms):
    s = sorted(times_ms)
    return {
        "mean_ms": statistics.fmean(s),
        "median_ms": statistics.median(s),
        "std_ms": statistics.pstdev(s),
        "min_ms": s[0],
        "p95_ms": s[int(0.95 * (len(s) - 1))],
        "n": len(s),
    }


def time_fn(fn):
    """Warm up, then time each call separately (GPU-synchronized)."""
    for _ in range(NUM_WARMUP):
        fn()
    sync()
    gc.collect()
    gc.disable()  # avoid garbage-collector pauses inside the timed region
    times = []
    try:
        for _ in range(NUM_ITERATIONS):
            sync()
            t0 = time.perf_counter()
            fn()
            sync()
            times.append((time.perf_counter() - t0) * 1000)
    finally:
        gc.enable()
    return times


def random_ids(length):
    """Fixed random token ids: reproducible, and latency does not depend on token values."""
    g = torch.Generator().manual_seed(SEED)
    ids = torch.randint(0, len(tokenizer), (BATCH_SIZE, length), generator=g)
    return ids.to(DEVICE)


def count_size(m):
    """Count each tensor once (shared/tied weights would otherwise be counted twice)."""
    seen, n_params, p_bytes = set(), 0, 0
    for p in m.parameters():
        key = (p.data_ptr(), p.numel())
        if key in seen:
            continue
        seen.add(key)
        n_params += p.numel()
        p_bytes += p.numel() * p.element_size()
    b_bytes = sum(b.numel() * b.element_size() for b in m.buffers())
    return n_params, p_bytes, b_bytes


# ============================================================
# FP4 E2M1 grid + rounding (element format inside NVFP4)
# ============================================================

FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
FP4_MAX = 6.0
FP4_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
FP32_TINY = torch.finfo(torch.float32).tiny

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max   # 448.0


def fp4_round_index(x):
    """
    Round |x| to the nearest FP4 level, ties to even. Returns the 3-bit magnitude code (0..7).
    bucketize keeps memory proportional to x (a full 8-way distance matrix would need
    several GB for this model's 50304x768 output layer).
    """
    a = x.abs()
    lo = torch.bucketize(a, FP4_MIDPOINTS, right=False)
    hi = torch.bucketize(a, FP4_MIDPOINTS, right=True)
    return torch.where((lo == hi) | (lo % 2 == 0), lo, hi)


_LUT_CACHE = {}


def get_lut256(device):
    """(256, 2) table: the two FP4 values packed into each byte. Shared, 2 KB, not model storage."""
    key = str(device)
    if key not in _LUT_CACHE:
        code_to_value = torch.cat([FP4_LEVELS, -FP4_LEVELS])   # codes 8..15 = negative
        b = torch.arange(256)
        lut = torch.stack([code_to_value[b & 0xF], code_to_value[b >> 4]], dim=1)
        _LUT_CACHE[key] = lut.to(device)
    return _LUT_CACHE[key]


# ============================================================
# NVFP4 quantized Linear
# ============================================================

class QuantizedLinearNVFP4(nn.Module):

    def __init__(self, linear):
        super().__init__()

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.n_weights = self.in_features * self.out_features

        if self.in_features % NV_BLOCK_SIZE != 0:
            raise ValueError(
                f"in_features ({self.in_features}) must be divisible by NV_BLOCK_SIZE ({NV_BLOCK_SIZE})"
            )
        self.n_blocks = self.in_features // NV_BLOCK_SIZE
        device = linear.weight.device

        # One-time quantization on CPU in FP32.
        w = linear.weight.detach().to("cpu", torch.float32).reshape(
            self.out_features, self.n_blocks, NV_BLOCK_SIZE
        )

        # ---- level 1: one FP32 scale for the whole tensor ----
        # Chosen so that the block containing the tensor's largest |weight|
        # maps to FP8_MAX before FP8 quantization, using the FP8 range fully.
        global_amax = w.abs().max().clamp(min=FP32_TINY)
        tensor_scale = (global_amax / (FP4_MAX * FP8_MAX)).item()   # python float, stored once

        # ---- level 2: per-block scale, quantized into real FP8 E4M3 ----
        block_amax = w.abs().amax(dim=-1, keepdim=True)
        ideal_block_scale = torch.clamp(block_amax / FP4_MAX, min=FP32_TINY)
        scale_for_fp8 = torch.clamp(ideal_block_scale / tensor_scale, max=FP8_MAX)
        block_scale_fp8 = scale_for_fp8.to(FP8_DTYPE)                 # actual stored format
        block_scale = block_scale_fp8.to(torch.float32) * tensor_scale  # scale really used to dequantize

        # ---- round weights to the FP4 grid using that two-level scale ----
        normalized = w / block_scale
        idx = fp4_round_index(normalized)                    # 0..7
        neg = (normalized < 0) & (idx != 0)
        q_value = FP4_LEVELS[idx] * torch.where(neg, -1.0, 1.0)

        # Quality number: how far the rebuilt weights are from the originals.
        self.sq_err = ((q_value * block_scale - w) ** 2).sum(dtype=torch.float64).item()
        self.sq_ref = (w ** 2).sum(dtype=torch.float64).item()

        # ---- store: packed 4-bit codes (uint8) + FP8 block scales + one FP32 tensor scale ----
        code = (idx | (neg.long() << 3)).reshape(self.out_features, self.in_features)
        packed = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8).contiguous()
        self.register_buffer("qweight", packed.to(device))                                  # (out, in/2) uint8
        self.register_buffer("block_scale_fp8", block_scale_fp8.squeeze(-1).contiguous().to(device))  # (out, n_blocks) fp8
        self.register_buffer("tensor_scale", torch.tensor(tensor_scale, dtype=torch.float32, device=device))  # scalar

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone())
        else:
            self.register_buffer("bias", None)

    def forward(self, x):
        lut = get_lut256(self.qweight.device)
        weight = torch.index_select(lut, 0, self.qweight.view(-1).to(torch.int32))
        weight = weight.view(self.out_features, self.n_blocks, NV_BLOCK_SIZE)

        block_scale = self.block_scale_fp8.to(torch.float32) * self.tensor_scale
        weight.mul_(block_scale.unsqueeze(-1))   # in place: no extra temporary

        return F.linear(x, weight.view(self.out_features, self.in_features), self.bias)


def quantize_model_nvfp4(m):
    for name, module in list(m.named_modules()):
        if isinstance(module, nn.Linear):
            parent = m
            *path, child_name = name.split(".")
            for part in path:
                parent = getattr(parent, part)
            setattr(parent, child_name, QuantizedLinearNVFP4(module))


# ============================================================
# Model loading
# ============================================================

print("PyTorch:", torch.__version__)
print("HIP:", torch.version.hip)
print("Device:", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=DTYPE,
    attn_implementation=ATTN_IMPL,
)
model = model.to(DEVICE)
model.eval()
model.requires_grad_(False)

print("Model loaded.")
print("Original dtype:", next(model.parameters()).dtype, "| attention:", ATTN_IMPL)


# ============================================================
# Reference numbers from the FP32 model (taken BEFORE quantizing)
# ============================================================

orig_params, orig_param_bytes, orig_buffer_bytes = count_size(model)
fp32_storage_bytes = orig_param_bytes + orig_buffer_bytes

check_ids = random_ids(64)
with torch.inference_mode():
    ref_logits = model(input_ids=check_ids, use_cache=False).logits.clone()


# ============================================================
# Quantize model
# ============================================================

print("\n--- NVFP4 QUANTIZATION ---")
print(f"Block size:        {NV_BLOCK_SIZE}")
print("Element format:    FP4 E2M1  {0, 0.5, 1, 1.5, 2, 3, 4, 6} + sign")
print(f"Block scale:       real FP8 E4M3 (max {FP8_MAX:g}), one per block")
print("Tensor scale:      one FP32 value per Linear layer")
print("Weight storage:    packed 4-bit codes in uint8 (2 per byte)")

t0 = time.perf_counter()
quantize_model_nvfp4(model)
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
model.eval()
model.requires_grad_(False)
print(f"NVFP4 quantization complete in {time.perf_counter() - t0:.1f} s.")

# What was quantized (embeddings and LayerNorm stay FP32, same as the other scripts).
q_modules = [m for m in model.modules() if isinstance(m, QuantizedLinearNVFP4)]
q_params = sum(m.n_weights for m in q_modules)
q_bytes = sum(
    m.qweight.numel() * m.qweight.element_size()
    + m.block_scale_fp8.numel() * m.block_scale_fp8.element_size()
    + m.tensor_scale.numel() * m.tensor_scale.element_size()   # 4 bytes, amortized over the whole layer
    for m in q_modules
)
bits_per_quantized_weight = 8 * q_bytes / q_params
weight_rel_rmse = math.sqrt(sum(m.sq_err for m in q_modules) / sum(m.sq_ref for m in q_modules))

print(f"Quantized layers:  {len(q_modules)} Linear layers, {q_params:,} weights")
print(f"Bits per quantized weight (incl. FP8 block scale + amortized tensor scale): {bits_per_quantized_weight:.4f}")
print("  (spec value ~4.5 bits/weight as tensors grow large; small layers pay more for the tensor scale)")
print(f"Weight relative RMSE: {weight_rel_rmse:.4e}")

# Sanity check: compare NVFP4 output against the original FP32 model.
# NVFP4's real FP8 block scale should reconstruct weights more closely than MXFP4's
# power-of-two-only scale, at a very similar bit cost.
with torch.inference_mode():
    q_logits = model(input_ids=check_ids, use_cache=False).logits
logit_diff = (q_logits - ref_logits).abs()
max_logit_diff = logit_diff.max().item()
mean_logit_diff = logit_diff.mean().item()
print(f"Max  |logit diff| vs FP32: {max_logit_diff:.3e}")
print(f"Mean |logit diff| vs FP32: {mean_logit_diff:.3e}")
del ref_logits, q_logits, logit_diff
gc.collect()


# ============================================================
# Model size
# ============================================================

_, param_bytes, buffer_bytes = count_size(model)
total_bytes = param_bytes + buffer_bytes
num_params = orig_params

print("\n--- MODEL SIZE ---")
print(f"Parameters:        {num_params:,}")
print(f"Parameter memory:  {param_bytes / MB:.2f} MB   (embeddings / norms left in FP32)")
print(f"Buffer memory:     {buffer_bytes / MB:.2f} MB   (NVFP4 codes + FP8 block scales + tensor scales + rotary etc.)")
print(f"Total storage:     {total_bytes / MB:.2f} MB")
print(f"FP32 reference:    {fp32_storage_bytes / MB:.2f} MB")
print(f"Size vs FP32:      {total_bytes / fp32_storage_bytes:.4f}x")
print(f"Bits per param:    {8 * total_bytes / num_params:.2f}   (whole model)")


# ============================================================
# Prefill latency (full prompt in one forward pass)
# ============================================================

print("\n--- PREFILL LATENCY ---")
prefill_results = {}

with torch.inference_mode():
    for L in PREFILL_LENGTHS:
        ids = random_ids(L)
        stats = summarize(time_fn(lambda: model(input_ids=ids, use_cache=False)))
        stats["tokens_per_sec"] = BATCH_SIZE * L / (stats["median_ms"] / 1000)
        prefill_results[L] = stats
        print(
            f"len={L:<4} median={stats['median_ms']:8.3f} ms  mean={stats['mean_ms']:8.3f} ms  "
            f"std={stats['std_ms']:6.3f}  p95={stats['p95_ms']:8.3f}  "
            f"{stats['tokens_per_sec']:10.1f} tok/s"
        )


# ============================================================
# Decode latency (one token at a time with KV cache)
# ============================================================

print("\n--- DECODE LATENCY (KV cache) ---")

decode_prompt = random_ids(DECODE_PROMPT_LEN)


def decode_once():
    step_ms = []
    out = model(input_ids=decode_prompt, use_cache=True)
    past = out.past_key_values
    nxt = out.logits[:, -1:].argmax(dim=-1)
    for _ in range(DECODE_NEW_TOKENS):
        sync()
        t0 = time.perf_counter()
        out = model(input_ids=nxt, past_key_values=past, use_cache=True)
        nxt = out.logits[:, -1:].argmax(dim=-1)
        past = out.past_key_values
        sync()
        step_ms.append((time.perf_counter() - t0) * 1000)
    return step_ms


with torch.inference_mode():
    for _ in range(2):  # warmup runs
        decode_once()
    gc.collect()
    gc.disable()
    all_steps = []
    try:
        for _ in range(DECODE_RUNS):
            all_steps.extend(decode_once())
    finally:
        gc.enable()

decode_stats = summarize(all_steps)
decode_stats["tokens_per_sec"] = BATCH_SIZE / (decode_stats["median_ms"] / 1000)
print(
    f"median={decode_stats['median_ms']:.3f} ms/token  mean={decode_stats['mean_ms']:.3f}  "
    f"std={decode_stats['std_ms']:.3f}  p95={decode_stats['p95_ms']:.3f}  "
    f"{decode_stats['tokens_per_sec']:.1f} tok/s"
)


# ============================================================
# Memory (measured on the longest prefill)
# ============================================================

print("\n--- MEMORY ---")
mem_ids = random_ids(max(PREFILL_LENGTHS))
memory_results = {}

if DEVICE == "cuda":
    gc.collect()
    torch.cuda.empty_cache()
    sync()
    baseline = torch.cuda.memory_allocated()   # weights only
    torch.cuda.reset_peak_memory_stats()       # reset AFTER empty_cache, BEFORE the forward
    with torch.inference_mode():
        _ = model(input_ids=mem_ids, use_cache=False)
    sync()
    peak = torch.cuda.max_memory_allocated()
    memory_results = {
        "weights_mb": baseline / MB,
        "peak_mb": peak / MB,
        "activation_mb": (peak - baseline) / MB,
    }
    print(f"Weights on device: {baseline / MB:.2f} MB")
    print(f"Peak during fwd:   {peak / MB:.2f} MB")
    print(f"Activation memory: {(peak - baseline) / MB:.2f} MB  (includes temporary dequantized weights)")
else:
    try:
        import psutil

        rss = psutil.Process().memory_info().rss
        memory_results = {"process_rss_mb": rss / MB}
        print(f"Process RSS:       {rss / MB:.2f} MB  (includes Python/torch overhead)")
    except ImportError:
        print("CPU run: `pip install psutil` to report process memory.")


# ============================================================
# Perplexity
# ============================================================

print("\n--- PERPLEXITY ---")

BUILTIN_TEXT = """
Artificial intelligence is a field of computer science concerned
with building systems that can perform tasks involving reasoning,
learning, perception, language understanding, and decision making.
Machine learning methods allow computers to identify patterns in
data and use those patterns to make predictions on previously unseen
examples.

Neural networks are computational models composed of interconnected
layers of mathematical operations. During training, the parameters
of a neural network are adjusted so that its predictions become
closer to the desired outputs. Modern language models use large
neural networks based on the Transformer architecture. These models
process sequences of tokens and learn statistical relationships
between words, symbols, and other elements of text.

The Transformer architecture uses attention mechanisms to determine
which parts of an input sequence are relevant to each other. This
allows information from different positions in a sequence to be
combined efficiently. During language-model training, the model is
typically given a sequence of tokens and trained to predict the next
token. Repeating this process over a large collection of documents
allows the model to learn linguistic structure and many statistical
regularities of written language.

Language models are commonly evaluated using loss and perplexity.
Cross-entropy loss measures how well a model predicts the observed
sequence of tokens. Perplexity is obtained by exponentiating the
average loss. Lower perplexity indicates that the model assigns
higher probability to the observed text.

Quantization reduces the numerical precision used to represent model
parameters. Instead of storing every parameter using a 32-bit
floating-point value, a quantized model can represent values using
fewer bits together with additional scale information. Four-bit
quantization is particularly useful because it can substantially
reduce storage requirements while attempting to preserve the behavior
of the original model.

Different quantization algorithms make different tradeoffs between
memory usage and reconstruction error. Scalar quantization assigns
values to a finite collection of reconstruction levels. Vector
quantization considers groups of values together. Rotations can
sometimes make the distribution of values more suitable for
quantization by spreading information more uniformly across
coordinates.

Memory consumption is another important consideration when evaluating
quantized neural networks. The theoretical number of bits per
parameter does not necessarily equal the physical memory consumed by
an implementation. Metadata, scales, lookup tables, padding, and
tensor data types can all contribute to actual storage.

Inference speed can also change after quantization. A low-bit
representation does not automatically make inference faster if the
implementation must repeatedly dequantize weights or perform
additional transformations. Optimized low-bit kernels can exploit
compressed representations directly, while a reference implementation
may reconstruct floating-point weights before each matrix
multiplication.

For this reason, quantization experiments should distinguish between
the properties of the quantization algorithm and the performance of
the particular implementation used for testing. Memory usage,
reconstruction error, model loss, perplexity, and inference latency
can each provide different information about the behavior of a
quantized model.

Modern language models are trained on large collections of documents
covering many different subjects. The resulting models can generate
text, answer questions, summarize information, translate between
languages, and perform other tasks involving natural language.

Efficient inference is important because neural networks contain
large numbers of parameters and require substantial computation.
Reducing the precision of these parameters can decrease memory
requirements, but the effect on accuracy depends on the quantization
method, the distribution of model weights, and the amount of
additional information stored alongside the quantized representation.

A useful benchmark therefore keeps the model architecture,
evaluation text, tokenizer, sequence length, and inference procedure
consistent across all experiments. This makes it possible to compare
different numerical representations while minimizing unrelated
sources of variation.
"""

if USE_WIKITEXT:
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    eval_text = "\n\n".join(ds["text"])
    eval_name = "wikitext-2-raw-v1 (test)"
else:
    eval_text = BUILTIN_TEXT
    eval_name = "built-in text"

eval_ids = tokenizer(eval_text, return_tensors="pt")["input_ids"].to(DEVICE)
eval_tokens = eval_ids.shape[1]

total_nll, total_pred = 0.0, 0
with torch.inference_mode():
    for start in range(0, eval_tokens, MAX_EVAL_LEN):
        chunk = eval_ids[:, start:start + MAX_EVAL_LEN]
        if chunk.shape[1] < 2:
            break
        loss = model(input_ids=chunk, labels=chunk, use_cache=False).loss.item()
        n_pred = chunk.shape[1] - 1
        total_nll += loss * n_pred
        total_pred += n_pred

avg_loss = total_nll / total_pred
perplexity = math.exp(avg_loss)

print(f"Dataset:           {eval_name}")
print(f"Tokens in text:    {eval_tokens}")
print(f"Tokens predicted:  {total_pred}")
print(f"Loss:              {avg_loss:.6f}")
print(f"Perplexity:        {perplexity:.6f}")


# ============================================================
# Summary + save
# ============================================================

print("\n--- NVFP4 SUMMARY ---")
print(f"Block size:        {NV_BLOCK_SIZE}")
print(f"Parameters:        {num_params:,}")
print(f"Model storage:     {total_bytes / MB:.2f} MB  ({total_bytes / fp32_storage_bytes:.4f}x FP32)")
print(f"Bits/quant weight: {bits_per_quantized_weight:.4f}")
for L, s in prefill_results.items():
    print(f"Prefill len={L:<4}  {s['median_ms']:.3f} ms  ({s['tokens_per_sec']:.1f} tok/s)")
print(f"Decode:            {decode_stats['median_ms']:.3f} ms/token  ({decode_stats['tokens_per_sec']:.1f} tok/s)")
if "peak_mb" in memory_results:
    print(f"Peak GPU memory:   {memory_results['peak_mb']:.2f} MB")
elif "process_rss_mb" in memory_results:
    print(f"Process RSS:       {memory_results['process_rss_mb']:.2f} MB")
print(f"Loss / PPL:        {avg_loss:.6f} / {perplexity:.6f}")
print(f"Weight rel. RMSE:  {weight_rel_rmse:.4e}")
print(f"Max |logit diff|:  {max_logit_diff:.3e}  (vs FP32)")

results = {
    "env": {
        "model": MODEL_NAME,
        "method": "custom_fake_nvfp4",
        "dtype": str(DTYPE),
        "attention": ATTN_IMPL,
        "device": DEVICE,
        "device_name": torch.cuda.get_device_name(0) if DEVICE == "cuda" else platform.processor(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "cpu_threads": torch.get_num_threads(),
        "batch_size": BATCH_SIZE,
        "warmup": NUM_WARMUP,
        "iterations": NUM_ITERATIONS,
    },
    "quantization": {
        "format": "NVFP4 (NVIDIA microscaling, FP4 E2M1 elements)",
        "block_size": NV_BLOCK_SIZE,
        "fp4_levels": FP4_LEVELS.tolist(),
        "fp4_max": FP4_MAX,
        "block_scale_format": "float8_e4m3fn",
        "fp8_max": FP8_MAX,
        "tensor_scale_format": "torch.float32 (one per Linear layer)",
        "weight_dtype": "torch.uint8 (two 4-bit codes per byte)",
        "quantized_layers": len(q_modules),
        "quantized_params": q_params,
        "bits_per_quantized_weight": bits_per_quantized_weight,
        "weight_relative_rmse": weight_rel_rmse,
        "max_logit_diff_vs_fp32": max_logit_diff,
        "mean_logit_diff_vs_fp32": mean_logit_diff,
    },
    "size": {
        "params": num_params,
        "parameter_memory_mb": param_bytes / MB,
        "buffer_memory_mb": buffer_bytes / MB,
        "storage_mb": total_bytes / MB,
        "fp32_storage_mb": fp32_storage_bytes / MB,
        "bits_per_param": 8 * total_bytes / num_params,
    },
    "prefill": {str(k): v for k, v in prefill_results.items()},
    "decode": decode_stats,
    "memory": memory_results,
    "perplexity": {
        "dataset": eval_name,
        "tokens": eval_tokens,
        "tokens_predicted": total_pred,
        "loss": avg_loss,
        "ppl": perplexity,
    },
}

with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {RESULTS_FILE}")