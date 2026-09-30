"""
FP4 (E2M1) custom fake-quantization benchmark for a Hugging Face causal LM.

Pipeline:
    FP32 weights
        -> group-wise scaling (one FP32 scale per group), done once on CPU
        -> round to the FP4 E2M1 grid  {0, 0.5, 1, 1.5, 2, 3, 4, 6} with a sign bit
        -> store 4-bit codes packed two per byte (uint8)  + FP32 scale, as buffers
        -> on every forward: unpack with a lookup table, multiply by scale (FP32)
        -> FP32 F.linear

This is "fake" quantization: the Ryzen 7 8845HS CPU and the Radeon 780M have no FP4
math units, so FP4 is only a STORAGE format here. All math is FP32.

The benchmark harness (timing, memory, perplexity) is IDENTICAL to benchmark_fp32.py /
benchmark_int32.py / benchmark_bf16.py / benchmark_fp8_e4m3.py, so the JSON results
can be compared directly. Only the quantization section and the FP4 metadata differ.
"""

import gc
import json
import math
import statistics
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# Configuration  (keep identical to benchmark_fp32.py)
# ============================================================

MODEL_NAME = "EleutherAI/pythia-160m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # ROCm also shows up as "cuda"
DTYPE = torch.float32
ATTN_IMPL = "sdpa"
SEED = 0

GROUP_SIZE = 64                    # quantization group size (must divide in_features)

BATCH_SIZE = 1
NUM_WARMUP = 10
NUM_ITERATIONS = 50

PREFILL_LENGTHS = [32, 128, 512]
DECODE_PROMPT_LEN = 32
DECODE_NEW_TOKENS = 64
DECODE_RUNS = 5

MAX_EVAL_LEN = 1024

# ---- FP4-specific options ----
# "pow2"   : scale = smallest power of two with absmax/scale <= 6   (MX-style scale)
# "absmax" : scale = absmax / 6                                     (uses the FP4 range fully)
SCALE_MODE = "pow2"
# True  : real 4-bit storage (two codes per uint8). Memory numbers are honest.
# False : FP4 values kept as FP32 numbers (no memory saving; storage will be > FP32).
PACK_FP4 = True

RESULTS_FILE = "benchmark_fp4_e2m1.json"   # change the name if you run several variants

assert SCALE_MODE in ("pow2", "absmax")


# ============================================================
# Reproducibility / true-FP32 settings
# ============================================================

torch.manual_seed(SEED)
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False



def sync():
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
        "p95_ms": s[int(0.95 * (len(s) - 1))],
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
# FP4 E2M1 grid, rounding, packing
# ============================================================

# E2M1 = 1 sign, 2 exponent, 1 mantissa bit. Positive magnitudes by 3-bit code 0..7:
FP4_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
FP4_MAX = 6.0
# Halfway points between neighbouring levels (used to find the nearest level in O(n)).
FP4_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
FP32_TINY = torch.finfo(torch.float32).tiny


def fp4_round_index(x):
    """
    Round |x| to the nearest FP4 level, ties to even (the usual hardware rule).
    Returns the 3-bit magnitude code (0..7). Values above 6 saturate to 6.

    Uses bucketize (memory ~ size of x) instead of a distance matrix against all 8 levels,
    which would need 8x more memory (several GB for the 50304x768 output layer).
    """
    a = x.abs()
    lo = torch.bucketize(a, FP4_MIDPOINTS, right=False)   # on an exact tie: lower level
    hi = torch.bucketize(a, FP4_MIDPOINTS, right=True)    # on an exact tie: upper level
    # On a tie the two candidate codes are neighbours, so exactly one is even.
    return torch.where((lo == hi) | (lo % 2 == 0), lo, hi)


_LUT_CACHE = {}


def get_lut256(device):
    """
    (256, 2) table: for each byte, the two FP4 values stored in it
    (low nibble = even position, high nibble = odd position).
    Shared by all layers; it is only 2 KB, so it is not counted as model storage.
    """
    key = str(device)
    if key not in _LUT_CACHE:
        code_to_value = torch.cat([FP4_LEVELS, -FP4_LEVELS])   # codes 8..15 = negative
        b = torch.arange(256)
        lut = torch.stack([code_to_value[b & 0xF], code_to_value[b >> 4]], dim=1)
        _LUT_CACHE[key] = lut.to(device)
    return _LUT_CACHE[key]


# ============================================================
# FP4 quantized Linear
# ============================================================

class QuantizedLinearFP4(nn.Module):

    def __init__(self, linear):
        super().__init__()

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.n_weights = self.in_features * self.out_features

        if self.in_features % GROUP_SIZE != 0:
            raise ValueError(
                f"in_features ({self.in_features}) must be divisible by GROUP_SIZE ({GROUP_SIZE})"
            )
        self.n_groups = self.in_features // GROUP_SIZE
        device = linear.weight.device

        # One-time quantization on CPU in FP32 (safe on any GPU, cost is small).
        # FP32 is enough: dividing by a power of two is exact, and FP4 is far coarser than FP32.
        w = linear.weight.detach().to("cpu", torch.float32).reshape(
            self.out_features, self.n_groups, GROUP_SIZE
        )

        # ---- per-group scale ----
        amax = w.abs().amax(dim=-1, keepdim=True)
        target = torch.clamp(amax / FP4_MAX, min=FP32_TINY)
        if SCALE_MODE == "pow2":
            # Smallest power of two >= target, computed exactly with frexp (no log2 rounding).
            mant, exp = torch.frexp(target)                 # target = mant * 2**exp, mant in [0.5, 1)
            exp = exp - (mant == 0.5).to(exp.dtype)         # target is already an exact power of two
            scale = torch.ldexp(torch.ones_like(target), exp)
        else:
            scale = target

        # ---- round to the FP4 grid ----
        normalized = w / scale
        idx = fp4_round_index(normalized)                    # 0..7
        neg = (normalized < 0) & (idx != 0)
        q_value = FP4_LEVELS[idx] * torch.where(neg, -1.0, 1.0)

        # Quality number: how far the rebuilt weights are from the originals.
        self.sq_err = ((q_value * scale - w) ** 2).sum(dtype=torch.float64).item()
        self.sq_ref = (w ** 2).sum(dtype=torch.float64).item()

        # ---- store ----
        if PACK_FP4:
            code = (idx | (neg.long() << 3)).reshape(self.out_features, self.in_features)
            packed = (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8).contiguous()
            self.register_buffer("qweight", packed.to(device))           # (out, in/2) uint8
        else:
            q_fp32 = q_value.reshape(self.out_features, self.in_features).contiguous()
            self.register_buffer("qweight", q_fp32.to(device))           # (out, in) fp32

        self.register_buffer("weight_scale", scale.squeeze(-1).contiguous().to(device))  # (out, n_groups)

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone())
        else:
            self.register_buffer("bias", None)

    def forward(self, x):
        if PACK_FP4:
            # byte -> (low value, high value) with one table lookup, straight to FP32.
            lut = get_lut256(self.qweight.device)
            weight = torch.index_select(lut, 0, self.qweight.view(-1).to(torch.int32))
            weight = weight.view(self.out_features, self.n_groups, GROUP_SIZE)
            weight.mul_(self.weight_scale.unsqueeze(-1))     # in place: no extra temporary
        else:
            weight = self.qweight.view(self.out_features, self.n_groups, GROUP_SIZE) \
                * self.weight_scale.unsqueeze(-1)

        return F.linear(x, weight.view(self.out_features, self.in_features), self.bias)


def quantize_model_fp4(m):
    for name, module in list(m.named_modules()):
        if isinstance(module, nn.Linear):
            parent = m
            *path, child_name = name.split(".")
            for part in path:
                parent = getattr(parent, part)
            setattr(parent, child_name, QuantizedLinearFP4(module))


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

# After quantization the weights become buffers, so model.parameters() no longer
# counts them. We save the true parameter count and FP32 size now.
orig_params, orig_param_bytes, orig_buffer_bytes = count_size(model)
fp32_storage_bytes = orig_param_bytes + orig_buffer_bytes

check_ids = random_ids(64)
with torch.inference_mode():
    ref_logits = model(input_ids=check_ids, use_cache=False).logits.clone()


# ============================================================
# Quantize model
# ============================================================

print("\n--- FP4 (E2M1) QUANTIZATION ---")
print(f"Group size:        {GROUP_SIZE}")
print("Levels:            0, 0.5, 1, 1.5, 2, 3, 4, 6 (+ sign)")
print(f"Scale mode:        {SCALE_MODE}  (scale stored as FP32)")
print(f"Weight storage:    {'packed 4-bit codes in uint8' if PACK_FP4 else 'FP32 values (no memory saving)'}")

t0 = time.perf_counter()
quantize_model_fp4(model)
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
model.eval()
model.requires_grad_(False)
print(f"FP4 quantization complete in {time.perf_counter() - t0:.1f} s.")

# What was quantized (embeddings and LayerNorm stay FP32, same as the other scripts).
q_modules = [m for m in model.modules() if isinstance(m, QuantizedLinearFP4)]
q_params = sum(m.n_weights for m in q_modules)
q_bytes = sum(
    m.qweight.numel() * m.qweight.element_size()
    + m.weight_scale.numel() * m.weight_scale.element_size()
    for m in q_modules
)
bits_per_quantized_weight = 8 * q_bytes / q_params
weight_rel_rmse = math.sqrt(sum(m.sq_err for m in q_modules) / sum(m.sq_ref for m in q_modules))

print(f"Quantized layers:  {len(q_modules)} Linear layers, {q_params:,} weights")
print(f"Bits per quantized weight (incl. scales): {bits_per_quantized_weight:.3f}")
print(f"Weight relative RMSE: {weight_rel_rmse:.4e}")

# Sanity check: compare FP4 output against the original FP32 model.
# FP4 has a very coarse grid, so large error is expected.
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
num_params = orig_params   # logical parameter count of the model

print("\n--- MODEL SIZE ---")
print(f"Parameters:        {num_params:,}")
print(f"Parameter memory:  {param_bytes / MB:.2f} MB   (embeddings / norms left in FP32)")
print(f"Buffer memory:     {buffer_bytes / MB:.2f} MB   (FP4 weights + scales + rotary etc.)")
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

gc.collect()
torch.cuda.empty_cache()
sync()
baseline = torch.cuda.memory_allocated()   # model memory before forward
torch.cuda.reset_peak_memory_stats()       # reset AFTER empty_cache, BEFORE the forward
with torch.inference_mode():
    _ = model(input_ids=mem_ids, use_cache=False)
sync()
peak = torch.cuda.max_memory_allocated()
memory_results = {
    "model_before_fwd_mb": baseline / MB,
    "peak_mb": peak / MB,
    "activation_mb": (peak - baseline) / MB,
}
print(f"Model memory before forward: {baseline / MB:.2f} MB")
print(f"Peak during fwd:   {peak / MB:.2f} MB")
print(f"Activation memory: {(peak - baseline) / MB:.2f} MB  (includes temporary dequantized weights)")


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


eval_text = BUILTIN_TEXT
eval_name = "built-in text"

eval_ids = tokenizer(eval_text, return_tensors="pt")["input_ids"].to(DEVICE)
eval_tokens = eval_ids.shape[1]

# Non-overlapping windows; each window predicts (len - 1) tokens,
# weighted by that count so the average is exact.
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

print("\n--- FP4 (E2M1) SUMMARY ---")
print(f"Group size:        {GROUP_SIZE}   scale mode: {SCALE_MODE}   packed: {PACK_FP4}")
print(f"Parameters:        {num_params:,}")
print(f"Model storage:     {total_bytes / MB:.2f} MB  ({total_bytes / fp32_storage_bytes:.4f}x FP32)")
print(f"Bits/quant weight: {bits_per_quantized_weight:.3f}")
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
        "method": "custom_fake_fp4_e2m1",
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
        "bits": 4,
        "format": "FP4 E2M1",
        "fp4_levels": FP4_LEVELS.tolist(),
        "fp4_max": FP4_MAX,
        "group_size": GROUP_SIZE,
        "scale_mode": SCALE_MODE,
        "packed": PACK_FP4,
        "weight_dtype": "torch.uint8 (two 4-bit codes per byte)" if PACK_FP4 else "torch.float32",
        "scale_dtype": "torch.float32",
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