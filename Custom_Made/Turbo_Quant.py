"""
TurboQuant custom fake-quantization benchmark for a Hugging Face causal LM.

TurboQuant, per your original script:
    FP32 block
        -> L2-normalize (radius r, stored FP16 + unit direction)
        -> randomized Walsh-Hadamard rotation (decorrelates the direction's coordinates)
        -> nearest-centroid lookup in a 16-point Lloyd-Max codebook fit to the
           Beta-distributed marginal of a rotated coordinate
        -> on every forward: centroid lookup -> inverse rotation -> * radius
        -> FP32 F.linear

This version fixes two real problems in the naive script and brings the harness in
line with the rest of this series (FP32 / INT32 / INT16 / INT4 / BF16 / FP8 E4M3 /
FP4 E2M1 / MXFP4 / NVFP4 / PolarQuant), so all the JSON outputs are comparable:

  1. STORAGE: NUM_LEVELS=16 needs only 4 bits, but the naive script stored each code
     as a full uint8 (8 bits) -- exactly 2x too much. Codes are now packed 2/byte,
     same trick used in the FP4/INT4/PolarQuant scripts. BLOCK_SIZE=128 is even, so
     this packs cleanly with no padding needed.

  2. SHARED-TABLE DOUBLE COUNTING: the naive script called self.register_buffer for
     the centroids and rotation signs INSIDE every quantized layer, so the SAME
     16-entry codebook and 128-entry sign vector got duplicated once per layer
     (~73 layers in this model). They are genuinely shared, global, read-only
     tables, so they are now built once, moved to the device once, and referenced
     by every layer -- not registered as a buffer -- with their one-off memory cost
     reported separately instead of inflating every layer's count.

This is "fake" quantization: it measures the FORMAT, not real hardware speed.
"""

import gc
import json
import math
import statistics
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# Configuration  (keep identical to benchmark_fp32.py)
# ============================================================

MODEL_NAME = "EleutherAI/pythia-160m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32
ATTN_IMPL = "sdpa"                 # your naive script used "eager" (the slow reference)
SEED = 0

BLOCK_SIZE = 128
BITS = 4
NUM_LEVELS = 2 ** BITS
ROTATION_SEED = 46
LLOYD_MAX_ITERS = 100

BATCH_SIZE = 1
NUM_WARMUP = 10
NUM_ITERATIONS = 50

PREFILL_LENGTHS = [32, 128, 512]
DECODE_PROMPT_LEN = 32
DECODE_NEW_TOKENS = 64
DECODE_RUNS = 5

MAX_EVAL_LEN = 1024
RESULTS_FILE = "benchmark_turboquant.json"


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
# Helpers (identical to the rest of the series)
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
# Normalized Walsh-Hadamard Transform (unchanged from your script)
# ============================================================

def fwht(x):
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError(f"FWHT size must be a power of two, got {n}")
    y = x.reshape(-1, n).clone()
    h = 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2)
        y = y.reshape(-1, n)
        h *= 2
    return y.reshape(x.shape) / math.sqrt(n)

def build_hadamard_matrix(n):
    """Precomputed n x n normalized Sylvester Hadamard matrix: H @ H == I, H symmetric."""
    H = torch.tensor([[1.0]])
    while H.shape[0] < n:
        H = torch.cat([
            torch.cat([H, H], dim=1),
            torch.cat([H, -H], dim=1),
        ], dim=0)
    return H / math.sqrt(n)

def build_random_signs(size, seed):
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=size)
    return torch.tensor(signs, dtype=torch.float32)


def turboquant_rotate(x, signs):
    signs = signs.to(device=x.device, dtype=x.dtype)
    return fwht(x * signs)


def turboquant_inverse_rotate(x, signs):
    signs = signs.to(device=x.device, dtype=x.dtype)
    return fwht(x) * signs


# ============================================================
# TurboQuant Beta distribution + Lloyd-Max codebook (unchanged math)
# ============================================================

def beta_pdf(x, dimension):
    exponent = (dimension - 3) / 2.0
    log_constant = (
        math.lgamma(dimension / 2.0)
        - 0.5 * math.log(math.pi)
        - math.lgamma((dimension - 1) / 2.0)
    )
    x = np.clip(x, -1.0 + 1e-12, 1.0 - 1e-12)
    return np.exp(log_constant + exponent * np.log(1.0 - x * x))


def build_lloyd_max_codebook(num_levels=16, dimension=128, iterations=100, grid_size=20001):
    x = np.linspace(-1.0 + 1e-7, 1.0 - 1e-7, grid_size, dtype=np.float64)
    pdf = beta_pdf(x, dimension)

    dx = x[1] - x[0]
    cdf = np.cumsum((pdf[:-1] + pdf[1:]) * 0.5 * dx)
    cdf = np.concatenate(([0.0], cdf))
    cdf /= cdf[-1]
    probabilities = (np.arange(num_levels) + 0.5) / num_levels
    centroids = np.interp(probabilities, cdf, x)

    for _ in range(iterations):
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        edges = np.concatenate(([-1.0], boundaries, [1.0]))
        new_centroids = np.empty_like(centroids)
        for i in range(num_levels):
            a, b = edges[i], edges[i + 1]
            mask = (x >= a) & (x <= b)
            x_region, pdf_region = x[mask], pdf[mask]
            denominator = np.trapezoid(pdf_region, x_region)
            numerator = np.trapezoid(x_region * pdf_region, x_region)
            new_centroids[i] = numerator / denominator
        difference = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids
        if difference < 1e-7:
            break
    return torch.tensor(centroids, dtype=torch.float32)


# Built once, GLOBAL, and referenced (not copied) by every quantized layer -- this
# is what fixes the per-layer duplication bug in the naive script.
ROTATION_SIGNS = build_random_signs(BLOCK_SIZE, ROTATION_SEED)
TURBOQUANT_CENTROIDS = build_lloyd_max_codebook(
    num_levels=NUM_LEVELS, dimension=BLOCK_SIZE, iterations=LLOYD_MAX_ITERS
)
HADAMARD_MATRIX = build_hadamard_matrix(BLOCK_SIZE)

print("\nTurboQuant rotation:")
print("Rotation:          Randomized Walsh-Hadamard")
print("Rotation seed:     ", ROTATION_SEED)
print("Centroids:         ", TURBOQUANT_CENTROIDS.tolist())


# ============================================================
# Uniform-index packing (2 codes per byte -- BLOCK_SIZE=128 is even, no padding needed)
# ============================================================

def pack_nibbles(code):
    """code: (..., n) int64 in [0, 15], n even -> packed uint8 (..., n/2)."""
    code = code.to(torch.uint8)
    return (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous()


def unpack_nibbles(packed):
    """Inverse of pack_nibbles. Returns 2x as many codes as the packed tensor's last dim."""
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    return torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)


# ============================================================
# TurboQuant quantized Linear
# ============================================================

class QuantizedLinearTurbo(nn.Module):

    def __init__(self, linear, centroids, rotation_signs, rotation_matrix_inv):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.n_weights = self.in_features * self.out_features

        if self.in_features % BLOCK_SIZE != 0:
            raise ValueError(
                f"in_features ({self.in_features}) must be divisible by BLOCK_SIZE ({BLOCK_SIZE})"
            )
        self.n_blocks = self.in_features // BLOCK_SIZE

        # Shared, read-only, global tables -- kept as plain references, NOT registered
        # buffers, so they are not duplicated once per layer.
        self.centroids = centroids
        self.rotation_signs = rotation_signs
        self.rotation_matrix_inv = rotation_matrix_inv

        weight = linear.weight.detach().to("cpu", torch.float32)
        weight_blocks = weight.reshape(self.out_features, self.n_blocks, BLOCK_SIZE)

        norm = torch.linalg.vector_norm(weight_blocks, dim=-1, keepdim=True).clamp(min=1e-8)
        normalized = weight_blocks / norm

        rotated = turboquant_rotate(normalized, rotation_signs.cpu())
        distances = (rotated.unsqueeze(-1) - centroids.cpu()).abs()
        idx = distances.argmin(dim=-1)                              # (out, n_blocks, D), 0..15

        # Quality number: how far the rebuilt weights are from the originals, using the
        # ACTUAL stored precision (FP16 norm), same care as the PolarQuant script.
        rotated_hat = centroids.cpu()[idx]
        normalized_hat = turboquant_inverse_rotate(rotated_hat, rotation_signs.cpu())
        stored_norm = norm.to(torch.float16).to(torch.float32)
        w_hat = normalized_hat * stored_norm
        self.sq_err = ((w_hat - weight_blocks) ** 2).sum(dtype=torch.float64).item()
        self.sq_ref = (weight_blocks ** 2).sum(dtype=torch.float64).item()

        self.register_buffer("qcodes", pack_nibbles(idx).to(linear.weight.device))
        self.register_buffer("block_norm", norm.squeeze(-1).to(torch.float16).to(linear.weight.device))

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone())
        else:
            self.register_buffer("bias", None)

    def forward(self, x):
        idx = unpack_nibbles(self.qcodes).long()                     # (out, n_blocks, D)
        rotated = self.centroids[idx]
        normalized = rotated @ self.rotation_matrix_inv
        weight_blocks = normalized * self.block_norm.float().unsqueeze(-1)
        weight = weight_blocks.reshape(self.out_features, self.in_features)
        return F.linear(x, weight, self.bias)


def quantize_model_turbo(m, centroids, rotation_signs, rotation_matrix_inv):
    for name, module in list(m.named_modules()):
        if isinstance(module, nn.Linear):
            parent = m
            *path, child_name = name.split(".")
            for part in path:
                parent = getattr(parent, part)
            layer = QuantizedLinearTurbo(module, 
                                         centroids, 
                                         rotation_signs, 
                                         rotation_matrix_inv).to(module.weight.device)
            setattr(parent, child_name, layer)


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

print("\n--- TURBOQUANT QUANTIZATION ---")
print(f"Block size:        {BLOCK_SIZE}")
print(f"Bits:              {BITS}  ({NUM_LEVELS} levels)")
print("Rotation:          Randomized Walsh-Hadamard")
print("Normalization:     L2 (radius stored FP16)")
print("Codebook:          Lloyd-Max Beta (16 shared FP32 centroids)")
print("Weight storage:    packed 4-bit codebook indices in uint8 (2 per byte)")

centroids_dev = TURBOQUANT_CENTROIDS.to(DEVICE)
rotation_signs_dev = ROTATION_SIGNS.to(DEVICE)
rotation_matrix_inv_dev = (HADAMARD_MATRIX * ROTATION_SIGNS.unsqueeze(0)).to(DEVICE)
shared_overhead_bytes = (
    centroids_dev.numel() * centroids_dev.element_size()
    + rotation_signs_dev.numel() * rotation_signs_dev.element_size()
    + rotation_matrix_inv_dev.numel() * rotation_matrix_inv_dev.element_size()
)   # counted ONCE, not per layer -- this is what the naive script got wrong

t0 = time.perf_counter()
quantize_model_turbo(model, centroids_dev, rotation_signs_dev, rotation_matrix_inv_dev)
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
model.eval()
model.requires_grad_(False)
print(f"TurboQuant quantization complete in {time.perf_counter() - t0:.1f} s.")

q_modules = [m for m in model.modules() if isinstance(m, QuantizedLinearTurbo)]
q_params = sum(m.n_weights for m in q_modules)
q_bytes = sum(
    m.qcodes.numel() * m.qcodes.element_size()
    + m.block_norm.numel() * m.block_norm.element_size()
    for m in q_modules
)
bits_per_quantized_weight = 8 * q_bytes / q_params
bits_per_quantized_weight_with_shared = 8 * (q_bytes + shared_overhead_bytes) / q_params
weight_rel_rmse = math.sqrt(sum(m.sq_err for m in q_modules) / sum(m.sq_ref for m in q_modules))

print(f"Quantized layers:  {len(q_modules)} Linear layers, {q_params:,} weights")
print(
    f"Shared table cost: {shared_overhead_bytes} bytes total "
    f"(centroids + rotation signs + inverse rotation matrix)"
)
print(f"Bits per quantized weight (codes + FP16 radius): {bits_per_quantized_weight:.4f}")
print(f"  (+ shared tables amortized over the whole model: {bits_per_quantized_weight_with_shared:.6f})")
print(f"Weight relative RMSE: {weight_rel_rmse:.4e}")

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
total_bytes = param_bytes + buffer_bytes + shared_overhead_bytes   # + the ONE shared copy
num_params = orig_params

print("\n--- MODEL SIZE ---")
print(f"Parameters:        {num_params:,}")
print(f"Parameter memory:  {param_bytes / MB:.2f} MB   (embeddings / norms left in FP32)")
print(f"Buffer memory:     {buffer_bytes / MB:.2f} MB   (TurboQuant codes + FP16 radii + other buffers)")
print(f"Shared tables:     {shared_overhead_bytes / MB:.4f} MB   (centroids + rotation signs, counted once)")
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

gc.collect()
torch.cuda.empty_cache()
sync()
baseline = torch.cuda.memory_allocated()
torch.cuda.reset_peak_memory_stats()
with torch.inference_mode():
    _ = model(input_ids=mem_ids, use_cache=False)
sync()
peak = torch.cuda.max_memory_allocated()
memory_results = {
    "model_memory_before_fwd_mb": baseline / MB,
    "peak_mb": peak / MB,
    "activation_mb": (peak - baseline) / MB,
}
print(f"Model memory before fwd: {baseline / MB:.2f} MB")
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

print("\n--- TURBOQUANT SUMMARY ---")
print(f"Block size:        {BLOCK_SIZE}   bits: {BITS}")
print(f"Parameters:        {num_params:,}")
print(f"Model storage:     {total_bytes / MB:.2f} MB  ({total_bytes / fp32_storage_bytes:.4f}x FP32)")
print(f"Bits/quant weight: {bits_per_quantized_weight:.4f}")
for L, s in prefill_results.items():
    print(f"Prefill len={L:<4}  {s['median_ms']:.3f} ms  ({s['tokens_per_sec']:.1f} tok/s)")
print(f"Decode:            {decode_stats['median_ms']:.3f} ms/token  ({decode_stats['tokens_per_sec']:.1f} tok/s)")
print(f"Peak GPU memory:   {memory_results['peak_mb']:.2f} MB")
print(f"Loss / PPL:        {avg_loss:.6f} / {perplexity:.6f}")
print(f"Weight rel. RMSE:  {weight_rel_rmse:.4e}")
print(f"Max |logit diff|:  {max_logit_diff:.3e}  (vs FP32)")

results = {
    "env": {
        "model": MODEL_NAME,
        "method": "custom_fake_turboquant",
        "dtype": str(DTYPE),
        "attention": ATTN_IMPL,
        "device": DEVICE,
        "device_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "batch_size": BATCH_SIZE,
        "warmup": NUM_WARMUP,
        "iterations": NUM_ITERATIONS,
    },
    "quantization": {
        "format": "TurboQuant (random Hadamard rotation + Lloyd-Max codebook)",
        "block_size": BLOCK_SIZE,
        "bits": BITS,
        "num_levels": NUM_LEVELS,
        "rotation_seed": ROTATION_SEED,
        "lloyd_max_iterations": LLOYD_MAX_ITERS,
        "centroids": TURBOQUANT_CENTROIDS.tolist(),
        "radius_dtype": "torch.float16",
        "code_dtype": "torch.uint8 (two 4-bit codes per byte)",
        "shared_table_bytes": shared_overhead_bytes,
        "quantized_layers": len(q_modules),
        "quantized_params": q_params,
        "bits_per_quantized_weight": bits_per_quantized_weight,
        "bits_per_quantized_weight_with_shared_tables": bits_per_quantized_weight_with_shared,
        "weight_relative_rmse": weight_rel_rmse,
        "max_logit_diff_vs_fp32": max_logit_diff,
        "mean_logit_diff_vs_fp32": mean_logit_diff,
    },
    "size": {
        "params": num_params,
        "parameter_memory_mb": param_bytes / MB,
        "buffer_memory_mb": buffer_bytes / MB,
        "shared_table_mb": shared_overhead_bytes / MB,
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