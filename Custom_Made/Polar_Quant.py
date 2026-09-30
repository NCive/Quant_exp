"""
PolarQuant custom fake-quantization benchmark for a Hugging Face causal LM.

PolarQuant idea: decompose each weight block into genuine POLAR (hyperspherical)
coordinates -- one radius (the block's L2 norm) and D-1 ANGLES that pin down the
direction on the unit sphere -- then quantize each angle UNIFORMLY.

This is the natural next step from your TurboQuant script, and the two are worth
contrasting directly:

                    TurboQuant                          PolarQuant
    ------------------------------------------------------------------------------
    Decorrelation   random Walsh-Hadamard rotation      none (works on raw block)
    Direction code  nearest-centroid lookup in a         explicit angle per axis,
                    Lloyd-Max codebook fit to the         quantized on a UNIFORM
                    Beta-distributed rotated coords       grid (no codebook, no
                                                           per-model fitting step)
    Extra storage   16 FP32 centroids + int8 rotation     none -- angles alone
                    signs, shared across all layers       reconstruct the direction

PolarQuant trades TurboQuant's optimal (but data-fitted) codebook for a fixed,
parameter-free uniform grid. Angles are not perfectly uniformly distributed even for
a random point on a sphere, so this is a deliberate simplicity-for-accuracy trade,
not a strict improvement -- the benchmark below is what tells you the size of that gap.

Pipeline (per block of BLOCK_SIZE weights):
    FP32 block
        -> norm r = ||block||_2  (stored as FP16, matching your TurboQuant script)
        -> unit vector u = block / r
        -> hyperspherical angles phi_1..phi_{D-2} in [0, pi], phi_{D-1} in [-pi, pi]
           (closed-form, vectorized with cumsum -- no Lloyd-Max fitting, no rotation)
        -> each angle uniformly quantized to BITS_PER_ANGLE bits, packed 2/byte
        -> on every forward: dequantize angles, rebuild u via cumprod of sines,
           multiply by r -> FP32 block
        -> FP32 F.linear

This is "fake" quantization, measuring the FORMAT, not real hardware speed.

The benchmark harness (timing, memory, perplexity) is IDENTICAL to the rest of this
series (FP32 / INT32 / INT16 / INT4 / BF16 / FP8 E4M3 / FP4 E2M1 / MXFP4 / NVFP4), so
the JSON results can be compared directly. Only the quantization section differs.
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
ATTN_IMPL = "sdpa"                 # your naive script used "eager" (the slow reference)
SEED = 0

BLOCK_SIZE = 128                   # same block size as your TurboQuant script
BITS_PER_ANGLE = 4                 # -> 16 uniform levels per angle
NUM_ANGLE_LEVELS = 2 ** BITS_PER_ANGLE

BATCH_SIZE = 1
NUM_WARMUP = 10
NUM_ITERATIONS = 50

PREFILL_LENGTHS = [32, 128, 512]
DECODE_PROMPT_LEN = 32
DECODE_NEW_TOKENS = 64
DECODE_RUNS = 5

MAX_EVAL_LEN = 1024
RESULTS_FILE = "benchmark_polarquant.json"


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
# Hyperspherical (polar) coordinates for a D-dim unit vector
# ============================================================
#
# For a unit vector u in R^D there are D-1 angles:
#   phi_1 .. phi_{D-2}  in [0, pi)   (standard "colatitude"-style angles)
#   phi_{D-1}            in [-pi, pi) (final angle, needs the full circle -> atan2)
#
#   u_1       = cos(phi_1)
#   u_k       = cos(phi_k) * prod_{j<k} sin(phi_j)      for k = 2 .. D-1
#   u_D       =              prod_{j<D-1} sin(phi_j)  (folds phi_{D-1}'s sign in)
#
# Both directions (Cartesian -> angles, angles -> Cartesian) are computed with
# cumsum/cumprod along the last axis -- no Python-level loop over the D-1 angles,
# so this is cheap enough to run on every forward call.

FP32_TINY = torch.finfo(torch.float32).tiny


def cartesian_to_angles(u):
    """u: (..., D) unit vectors -> phi: (..., D-1) angles."""
    D = u.shape[-1]
    sq = u * u
    # r2[..., k] = sum_{j=k}^{D-1} u_j^2  (reverse cumulative sum of squares)
    r2 = torch.flip(torch.cumsum(torch.flip(sq, dims=[-1]), dim=-1), dims=[-1])
    r = torch.sqrt(torch.clamp(r2, min=FP32_TINY))

    phi_body = torch.arccos(torch.clamp(u[..., :D - 2] / r[..., :D - 2], -1.0, 1.0))
    phi_last = torch.atan2(u[..., D - 1], u[..., D - 2]).unsqueeze(-1)
    return torch.cat([phi_body, phi_last], dim=-1)   # (..., D-1)


def angles_to_cartesian(phi, D):
    """phi: (..., D-1) angles -> u: (..., D) unit vectors."""
    cos_body = torch.cos(phi[..., :D - 2])
    sin_body = torch.sin(phi[..., :D - 2])
    cumprod_sin = torch.cumprod(sin_body, dim=-1)                      # inclusive
    excl_cumprod = torch.cat(
        [torch.ones_like(cumprod_sin[..., :1]), cumprod_sin[..., :-1]], dim=-1
    )
    u_body = cos_body * excl_cumprod                                    # (..., D-2)
    tail_prod = cumprod_sin[..., -1:] if D > 2 else torch.ones_like(phi[..., :1])
    u_second_last = torch.cos(phi[..., D - 2:D - 1]) * tail_prod
    u_last = torch.sin(phi[..., D - 2:D - 1]) * tail_prod
    return torch.cat([u_body, u_second_last, u_last], dim=-1)           # (..., D)


# ============================================================
# Uniform angle quantization + packing (2 codes per byte, like the FP4/INT4 scripts)
# ============================================================

def quantize_angles(phi, D):
    """phi: (..., D-1) -> uint8 codes 0..NUM_ANGLE_LEVELS-1, same shape."""
    body = phi[..., :D - 2] / math.pi                                  # [0, 1)
    last = (phi[..., D - 2:D - 1] + math.pi) / (2 * math.pi)           # [0, 1)
    unit = torch.cat([body, last], dim=-1)
    code = torch.round(unit * (NUM_ANGLE_LEVELS - 1)).clamp_(0, NUM_ANGLE_LEVELS - 1)
    return code.to(torch.int64)


def dequantize_angles(code, D):
    """Inverse of quantize_angles. code: int64/uint8 (..., D-1) -> phi (..., D-1)."""
    unit = code.to(torch.float32) / (NUM_ANGLE_LEVELS - 1)
    body = unit[..., :D - 2] * math.pi
    last = unit[..., D - 2:D - 1] * (2 * math.pi) - math.pi
    return torch.cat([body, last], dim=-1)


def pack_codes(code, n_angles):
    """
    code: (..., n_angles) int64 in [0, 15] -> packed uint8 (..., ceil(n_angles/2)).
    Pads with a dummy 0 code if n_angles is odd (costs nothing: packing already
    rounds up to a whole byte).
    """
    if n_angles % 2 == 1:
        pad = torch.zeros_like(code[..., :1])
        code = torch.cat([code, pad], dim=-1)
    code = code.to(torch.uint8)
    return (code[..., 0::2] | (code[..., 1::2] << 4)).contiguous()


def unpack_codes(packed, n_angles):
    """Inverse of pack_codes. Returns exactly n_angles codes (drops any padding)."""
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    interleaved = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], -1)
    return interleaved[..., :n_angles]

_ANGLE_LUT_CACHE = {}


def get_angle_luts(device):
    """
    (body_cos, body_sin, last_cos, last_sin), each length NUM_ANGLE_LEVELS, on `device`.
    Cached per device: each angle only takes NUM_ANGLE_LEVELS possible values after
    quantization, so cos()/sin() can be looked up instead of recomputed every forward call.
    """
    key = str(device)
    if key not in _ANGLE_LUT_CACHE:
        levels = torch.arange(NUM_ANGLE_LEVELS, dtype=torch.float32) / (NUM_ANGLE_LEVELS - 1)
        body_angle = levels * math.pi
        last_angle = levels * (2 * math.pi) - math.pi
        _ANGLE_LUT_CACHE[key] = (
            torch.cos(body_angle).to(device),
            torch.sin(body_angle).to(device),
            torch.cos(last_angle).to(device),
            torch.sin(last_angle).to(device),
        )
    return _ANGLE_LUT_CACHE[key]


def angles_to_cartesian_from_codes(code, D):
    """
    code: (..., D-1) integer codes in [0, NUM_ANGLE_LEVELS) -> u: (..., D) unit vectors.
    Same math as angles_to_cartesian(dequantize_angles(code, D), D), but reads cos/sin
    from a lookup table instead of calling torch.cos/torch.sin -- safe because a
    quantized angle can only take NUM_ANGLE_LEVELS distinct values.
    """
    body_cos_lut, body_sin_lut, last_cos_lut, last_sin_lut = get_angle_luts(code.device)
    body_code = code[..., :D - 2].long()   # .long() matters: uint8 used as an index
    last_code = code[..., D - 2:D - 1].long()   # is treated as a boolean mask, not an int index

    cos_body = body_cos_lut[body_code]
    sin_body = body_sin_lut[body_code]
    cos_last = last_cos_lut[last_code]
    sin_last = last_sin_lut[last_code]

    cumprod_sin = torch.cumprod(sin_body, dim=-1)
    excl_cumprod = torch.cat(
        [torch.ones_like(cumprod_sin[..., :1]), cumprod_sin[..., :-1]], dim=-1
    )
    u_body = cos_body * excl_cumprod
    tail_prod = cumprod_sin[..., -1:]
    u_second_last = cos_last * tail_prod
    u_last = sin_last * tail_prod
    return torch.cat([u_body, u_second_last, u_last], dim=-1)

# ============================================================
# PolarQuant quantized Linear
# ============================================================

class QuantizedLinearPolar(nn.Module):

    def __init__(self, linear):
        super().__init__()

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.n_weights = self.in_features * self.out_features

        if self.in_features % BLOCK_SIZE != 0:
            raise ValueError(
                f"in_features ({self.in_features}) must be divisible by BLOCK_SIZE ({BLOCK_SIZE})"
            )
        self.n_blocks = self.in_features // BLOCK_SIZE
        self.n_angles = BLOCK_SIZE - 1
        device = linear.weight.device

        # One-time quantization on CPU in FP32.
        w = linear.weight.detach().to("cpu", torch.float32).reshape(
            self.out_features, self.n_blocks, BLOCK_SIZE
        )

        norm = torch.linalg.vector_norm(w, dim=-1, keepdim=True).clamp(min=1e-8)
        u = w / norm
        phi = cartesian_to_angles(u)                       # (out, n_blocks, D-1)
        code = quantize_angles(phi, BLOCK_SIZE)             # (out, n_blocks, D-1), 0..15

        # Quality number: how far the rebuilt weights are from the originals.
        phi_hat = dequantize_angles(code, BLOCK_SIZE)
        u_hat = angles_to_cartesian(phi_hat, BLOCK_SIZE)

        # Actual stored radius is FP16, so include FP16 radius
        # quantization in the reconstruction-error measurement.
        stored_norm = norm.squeeze(-1).to(torch.float16).to(torch.float32).unsqueeze(-1)

        w_hat = u_hat * stored_norm
        self.sq_err = ((w_hat - w) ** 2).sum(dtype=torch.float64).item()
        self.sq_ref = (w ** 2).sum(dtype=torch.float64).item()

        packed = pack_codes(code, self.n_angles)            # (out, n_blocks, ceil(D-1 /2))
        self.register_buffer("qcodes", packed.contiguous().to(device))
        self.register_buffer("block_norm", norm.squeeze(-1).to(torch.float16).contiguous().to(device))

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone())
        else:
            self.register_buffer("bias", None)

    def forward(self, x):
        code = unpack_codes(self.qcodes, self.n_angles)                  # (out, n_blocks, D-1)
        u = angles_to_cartesian_from_codes(code, BLOCK_SIZE)              # (out, n_blocks, D)
        weight = u * self.block_norm.float().unsqueeze(-1)
        return F.linear(x, weight.view(self.out_features, self.in_features), self.bias)


def quantize_model_polar(m):
    for name, module in list(m.named_modules()):
        if isinstance(module, nn.Linear):
            parent = m
            *path, child_name = name.split(".")
            for part in path:
                parent = getattr(parent, part)
            setattr(parent, child_name, QuantizedLinearPolar(module))


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

print("\n--- POLARQUANT QUANTIZATION ---")
print(f"Block size:        {BLOCK_SIZE}")
print(f"Angles per block:  {BLOCK_SIZE - 1}  (+ 1 radius)")
print(f"Bits per angle:    {BITS_PER_ANGLE}  ({NUM_ANGLE_LEVELS} uniform levels)")
print("Rotation:          none")
print("Direction code:    uniform per-angle quantization (no codebook)")
print("Radius storage:    FP16")

t0 = time.perf_counter()
quantize_model_polar(model)
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
model.eval()
model.requires_grad_(False)
print(f"PolarQuant quantization complete in {time.perf_counter() - t0:.1f} s.")

# What was quantized (embeddings and LayerNorm stay FP32, same as the other scripts).
q_modules = [m for m in model.modules() if isinstance(m, QuantizedLinearPolar)]
q_params = sum(m.n_weights for m in q_modules)
q_bytes = sum(
    m.qcodes.numel() * m.qcodes.element_size()
    + m.block_norm.numel() * m.block_norm.element_size()
    for m in q_modules
)
bits_per_quantized_weight = 8 * q_bytes / q_params
weight_rel_rmse = math.sqrt(sum(m.sq_err for m in q_modules) / sum(m.sq_ref for m in q_modules))

print(f"Quantized layers:  {len(q_modules)} Linear layers, {q_params:,} weights")
print(f"Bits per quantized weight (codes + FP16 radius): {bits_per_quantized_weight:.4f}")
print(f"Weight relative RMSE: {weight_rel_rmse:.4e}")

# Sanity check: compare PolarQuant output against the original FP32 model.
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
print(f"Buffer memory:     {buffer_bytes / MB:.2f} MB   (PolarQuant codes + FP16 radii + other buffers)")
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
baseline = torch.cuda.memory_allocated()   # Model memory before fwd
torch.cuda.reset_peak_memory_stats()       # reset AFTER empty_cache, BEFORE the forward
with torch.inference_mode():
    _ = model(input_ids=mem_ids, use_cache=False)
sync()
peak = torch.cuda.max_memory_allocated()
memory_results = {
    "model_memory_before_fwd_mb": baseline / MB,
    "peak_mb": peak / MB,
    "activation_mb": (peak - baseline) / MB,
}
print(f"Model Memory before fwd: {baseline / MB:.2f} MB")
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

print("\n--- POLARQUANT SUMMARY ---")
print(f"Block size:        {BLOCK_SIZE}   bits/angle: {BITS_PER_ANGLE}")
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
        "method": "custom_fake_polarquant",
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
        "format": "PolarQuant (hyperspherical polar decomposition, uniform angle quantization)",
        "block_size": BLOCK_SIZE,
        "angles_per_block": BLOCK_SIZE - 1,
        "bits_per_angle": BITS_PER_ANGLE,
        "num_angle_levels": NUM_ANGLE_LEVELS,
        "radius_dtype": "torch.float16",
        "rotation": None,
        "direction_code": "uniform per-angle grid (no codebook)",
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