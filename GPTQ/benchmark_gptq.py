"""
GPTQ INT benchmark (gptqmodel) for pythia-160m. Single file.

Run order:
    1. python quantize_gptq.py        (BITS = BITS)
    2. python benchmark_gptq.py

Layout:
    Part 1  config + env vars      (edit this for each format)
    Part 2  benchmark harness      (identical for every format; do not edit)
    Part 3  load model + run       (edit the loading lines for each format)

Needs: pip install psutil datasets

Env vars MUST be set before torch / gptqmodel are imported, so they come first.
"""

import os

os.environ["GPTQ_TORCH_TRITON_DEQUANT"] = "0"   # eager PyTorch dequantization (no Triton)
os.environ["TORCH_CPP_LOG_LEVEL"] = "3"         # hide C++ warnings

import warnings

warnings.filterwarnings("ignore")

import time
from collections import Counter

import torch
from gptqmodel import GPTQModel, BACKEND
from transformers import AutoTokenizer

import gc
import json
import math
import statistics
import threading

import psutil
from transformers import AutoModelForCausalLM

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "EleutherAI/pythia-160m"

# BITS = 2
# GROUP_SIZE = 64

FP8_FORMAT = "float8_e5m2"        # "float8_e4m3fn" or "float8_e5m2" 
CHECKPOINT = f"pythia-160m-gptq-fp8-{FP8_FORMAT.split('_')[1]}"
RESULTS_FILE = f"benchmark_gptq-fp8-{FP8_FORMAT.split('_')[1]}.json"

# "cpu" for the Ryzen 7 8845HS itself, "cuda:0" if you run ROCm on the Radeon 780M.
DEVICE = torch.device(os.environ.get("BENCH_DEVICE", "cuda:0"))

# 8 physical cores. Using physical cores (not 16 SMT threads) is usually faster for matmul.
THREADS = int(os.environ.get("BENCH_THREADS", 8))

INFER_BACKEND = BACKEND.FP8_TORCH   # pinned on purpose, so results never depend on "AUTO" picking a kernel
EVAL_SOURCE = "builtin"              # "builtin" = same text as your earlier runs, "wikitext2" = standard PPL
EAGER_DEQUANT = True                 # keep TorchLinear out of torch.compile / Triton


# ============================================================
# Keep GPTQModel's TorchLinear in eager mode
# ============================================================
# GPTQModel calls TorchLinear.optimize() while loading. That wraps dequantization in
# torch.compile (Inductor / Triton), which needs a compiler toolchain. For a clean,
# reproducible benchmark we skip it. The checkpoint itself is unchanged.

# if EAGER_DEQUANT:
#     try:
#         from gptqmodel.nn_modules.qlinear.torch import TorchLinear

#         def optimize_eager(self, backend=None, mode=None, fullgraph=False):
#             if getattr(self, "optimized", False):
#                 return
#             self.optimized = True

#         TorchLinear.optimize = optimize_eager
#     except ImportError:
#         print("Note: TorchLinear not found in this gptqmodel version; eager patch skipped.")

if EAGER_DEQUANT:
    import importlib, pkgutil
    import gptqmodel.nn_modules.qlinear as _ql

    def optimize_eager(self, backend=None, mode=None, fullgraph=False):
        if getattr(self, "optimized", False):
            return
        self.optimized = True

    for _m in pkgutil.iter_modules(_ql.__path__):
        try:
            _mod = importlib.import_module(f"{_ql.__name__}.{_m.name}")
        except Exception:
            continue
        for _name, _cls in vars(_mod).items():
            if isinstance(_cls, type) and _name.endswith("Linear") and hasattr(_cls, "optimize"):
                _cls.optimize = optimize_eager



# ============================================================
# Settings (same values as your earlier FP32 / custom INT8 scripts)
# ============================================================

MB = 1024 ** 2
SEED = 0
BATCH_SIZE = 1

NUM_WARMUP = 10
NUM_ITERATIONS = 50

PREFILL_LENGTHS = [32, 128, 512]
DECODE_PROMPT_LEN = 32
DECODE_NEW_TOKENS = 64
DECODE_RUNS = 5

MAX_EVAL_LEN = 1024


# ============================================================
# Setup
# ============================================================

def setup(device, threads=None):
    """Fix seeds and force strict FP32 math so formats compare fairly."""
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if threads:
        torch.set_num_threads(threads)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


# ============================================================
# Small helpers
# ============================================================

def summarize(times_ms):
    s = sorted(times_ms)
    return {
        "mean_ms": statistics.fmean(s),
        "median_ms": statistics.median(s),
        "std_ms": statistics.pstdev(s),
        "p95_ms": s[int(0.95 * (len(s) - 1))],
    }


def time_fn(fn, device):
    """Warm up, then time every call separately."""
    for _ in range(NUM_WARMUP):
        fn()
    sync(device)
    gc.collect()
    gc.disable()  # no garbage-collector pauses inside the timed region
    times = []
    try:
        for _ in range(NUM_ITERATIONS):
            sync(device)
            t0 = time.perf_counter()
            fn()
            sync(device)
            times.append((time.perf_counter() - t0) * 1000)
    finally:
        gc.enable()
    return times


def random_ids(tokenizer, length, device):
    """Fixed random token ids: reproducible, latency does not depend on token values."""
    g = torch.Generator().manual_seed(SEED)
    ids = torch.randint(0, len(tokenizer), (BATCH_SIZE, length), generator=g)
    return ids.to(device)


def count_size(m):
    """Count each tensor once (tied weights would otherwise be counted twice)."""
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


def checkpoint_size_mb(path):
    import os
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / MB


# ============================================================
# FP32 reference (for accuracy checks)
# ============================================================

def fp32_reference(model_name, ids, device):
    """
    Load the original FP32 model, record its logits on `ids`, then free it.
    Returns (logits_on_cpu, fp32_storage_bytes, num_params).
    """
    ref = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device).eval()
    ref.requires_grad_(False)
    n_params, p_bytes, b_bytes = count_size(ref)
    with torch.inference_mode():
        logits = ref(input_ids=ids, use_cache=False).logits.float().cpu()
    del ref
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return logits, p_bytes + b_bytes, n_params


def fidelity(hf, ids, ref_logits):
    with torch.inference_mode():
        q = hf(input_ids=ids, use_cache=False).logits.float().cpu()
    diff = (q - ref_logits).abs()
    top1 = (q.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
    return {
        "max_abs_logit_diff": diff.max().item(),
        "mean_abs_logit_diff": diff.mean().item(),
        "top1_agreement": top1,
    }


# ============================================================
# Memory
# ============================================================

class _RSSPeak:
    """Samples process RSS in a background thread (CPU peak memory)."""

    def __init__(self, interval=0.002):
        self.proc = psutil.Process()
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self):
        self.peak = self.proc.memory_info().rss
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def measure_memory(fn, device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        sync(device)
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()  # after empty_cache, before forward
        fn()
        sync(device)
        peak = torch.cuda.max_memory_allocated()
        return {"kind": "cuda_allocated", "before_mb": base / MB,
                "peak_mb": peak / MB, "activation_mb": (peak - base) / MB}
    base = psutil.Process().memory_info().rss
    with _RSSPeak() as s:
        fn()
    return {"kind": "process_rss", "before_mb": base / MB,
            "peak_mb": s.peak / MB, "activation_mb": (s.peak - base) / MB}


# ============================================================
# Latency
# ============================================================

def prefill_benchmark(hf, tokenizer, device):
    out = {}
    with torch.inference_mode():
        for L in PREFILL_LENGTHS:
            ids = random_ids(tokenizer, L, device)
            st = summarize(time_fn(lambda: hf(input_ids=ids, use_cache=False), device))
            st["tokens_per_sec"] = BATCH_SIZE * L / (st["median_ms"] / 1000)
            out[str(L)] = st
            print(f"len={L:<4} median={st['median_ms']:9.3f} ms  mean={st['mean_ms']:9.3f}  "
                  f"std={st['std_ms']:7.3f}  p95={st['p95_ms']:9.3f}  {st['tokens_per_sec']:9.1f} tok/s")
    return out


def decode_benchmark(hf, tokenizer, device):
    prompt = random_ids(tokenizer, DECODE_PROMPT_LEN, device)

    def decode_once():
        steps = []
        o = hf(input_ids=prompt, use_cache=True)
        past = o.past_key_values
        nxt = o.logits[:, -1:].argmax(-1)
        for _ in range(DECODE_NEW_TOKENS):
            sync(device)
            t0 = time.perf_counter()
            o = hf(input_ids=nxt, past_key_values=past, use_cache=True)
            nxt = o.logits[:, -1:].argmax(-1)
            past = o.past_key_values
            sync(device)
            steps.append((time.perf_counter() - t0) * 1000)
        return steps

    with torch.inference_mode():
        for _ in range(2):
            decode_once()
        gc.collect()
        gc.disable()
        allsteps = []
        try:
            for _ in range(DECODE_RUNS):
                allsteps.extend(decode_once())
        finally:
            gc.enable()

    st = summarize(allsteps)
    st["tokens_per_sec"] = BATCH_SIZE / (st["median_ms"] / 1000)
    print(f"median={st['median_ms']:.3f} ms/token  mean={st['mean_ms']:.3f}  "
          f"std={st['std_ms']:.3f}  p95={st['p95_ms']:.3f}  {st['tokens_per_sec']:.1f} tok/s")
    return st


# ============================================================
# Perplexity
# ============================================================

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


def load_eval_text(source):
    """source = 'builtin' (matches your earlier runs) or 'wikitext2' (standard benchmark)."""
    if source == "wikitext2":
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return "\n\n".join(ds["text"]), "wikitext-2-raw-v1 (test)"
    return BUILTIN_TEXT, "built-in text"


def perplexity(hf, tokenizer, device, source="builtin"):
    text, name = load_eval_text(source)
    ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    n_tokens = ids.shape[1]

    total_nll, total_pred = 0.0, 0
    with torch.inference_mode():
        for start in range(0, n_tokens, MAX_EVAL_LEN):
            chunk = ids[:, start:start + MAX_EVAL_LEN]
            if chunk.shape[1] < 2:
                break
            loss = hf(input_ids=chunk, labels=chunk, use_cache=False).loss.item()
            n = chunk.shape[1] - 1  # tokens actually predicted in this window
            total_nll += loss * n
            total_pred += n

    avg = total_nll / total_pred
    return {"dataset": name, "tokens": n_tokens, "tokens_predicted": total_pred,
            "loss": avg, "ppl": math.exp(avg)}


# ============================================================
# One call runs everything
# ============================================================

def run_benchmark(hf, tokenizer, device, label, model_name, meta, results_file,
                  eval_source="builtin", fp32_storage_bytes=None, num_params=None,
                  fidelity_result=None, ckpt_mb=None):
    hf.eval()
    hf.requires_grad_(False)

    _, p_bytes, b_bytes = count_size(hf)
    total = p_bytes + b_bytes
    num_params = num_params or count_size(hf)[0]

    print(f"\n--- MODEL SIZE ({label}) ---")
    print(f"Parameter memory:  {p_bytes / MB:.2f} MB")
    print(f"Buffer memory:     {b_bytes / MB:.2f} MB")
    print(f"Total in memory:   {total / MB:.2f} MB")
    if fp32_storage_bytes:
        print(f"Size vs FP32:      {total / fp32_storage_bytes:.4f}x")
    print(f"Effective Bits/Param:    {8 * total / num_params:.2f}")
    if ckpt_mb:
        print(f"Checkpoint on disk: {ckpt_mb:.2f} MB")

    print("\n--- PREFILL LATENCY ---")
    prefill = prefill_benchmark(hf, tokenizer, device)

    print("\n--- DECODE LATENCY (KV cache) ---")
    decode = decode_benchmark(hf, tokenizer, device)

    print("\n--- MEMORY (longest prefill) ---")
    mem_ids = random_ids(tokenizer, max(PREFILL_LENGTHS), device)

    def fwd():
        with torch.inference_mode():
            hf(input_ids=mem_ids, use_cache=False)

    memory = measure_memory(fwd, device)
    print(f"Kind:              {memory['kind']}")
    print(f"Before forward:    {memory['before_mb']:.2f} MB")
    print(f"Peak:              {memory['peak_mb']:.2f} MB")
    print(f"Extra during fwd:  {memory['activation_mb']:.2f} MB")

    print("\n--- PERPLEXITY ---")
    ppl = perplexity(hf, tokenizer, device, eval_source)
    print(f"Dataset:           {ppl['dataset']}")
    print(f"Tokens predicted:  {ppl['tokens_predicted']}")
    print(f"Loss / PPL:        {ppl['loss']:.6f} / {ppl['ppl']:.6f}")

    if fidelity_result:
        print("\n--- FIDELITY vs FP32 ---")
        for k, v in fidelity_result.items():
            print(f"{k:<22} {v:.6g}")

    results = {
        "env": {
            "model": model_name, "method": label, "device": str(device),
            "threads": torch.get_num_threads(), "torch": torch.__version__,
            "batch_size": BATCH_SIZE, "warmup": NUM_WARMUP, "iterations": NUM_ITERATIONS,
            **meta,
        },
        "size": {"params": num_params, "storage_mb": total / MB,
                 "fp32_storage_mb": fp32_storage_bytes / MB if fp32_storage_bytes else None,
                 "effective_bits_per_param": 8 * total / num_params, "checkpoint_mb": ckpt_mb},
        "prefill": prefill, "decode": decode, "memory": memory,
        "perplexity": ppl, "fidelity": fidelity_result,
    }
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_file}")
    return results


# ============================================================
# Environment
# ============================================================

setup(DEVICE, THREADS)

# print(f"\n--- GPTQ INT{BITS} BENCHMARK ---")
print(f"\n--- GPTQ {FP8_FORMAT} BENCHMARK ---")
print("PyTorch:", torch.__version__)
print("Device:", DEVICE, "| threads:", torch.get_num_threads())
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0), "| HIP:", torch.version.hip)

if not os.path.isdir(CHECKPOINT):
    raise FileNotFoundError(f"GPTQ checkpoint not found: {CHECKPOINT}\nRun quantize_gptq.py first.")


# ============================================================
# FP32 reference (accuracy baseline), taken before loading the quantized model
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

check_ids = random_ids(tokenizer, 64, DEVICE)
ref_logits, fp32_bytes, num_params = fp32_reference(MODEL_NAME, check_ids, DEVICE)


# ============================================================
# Load quantized model
# ============================================================

print("\nLoading GPTQ model...")
t0 = time.perf_counter()
wrapper = GPTQModel.load(CHECKPOINT, device=str(DEVICE), backend=INFER_BACKEND)
load_time = time.perf_counter() - t0
print(f"Loaded in {load_time:.2f} s")

hf = wrapper.model          # plain Hugging Face model: same call style as the FP32 / custom scripts
# hf.float()                  # FP32 activations and scales, matching the other benchmarks

FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}
for mod in hf.modules():                      # cast to FP32, but leave FP8 weights alone
    for n, p in list(mod._parameters.items()):
        if p is not None and p.dtype.is_floating_point and p.dtype not in FP8_DTYPES:
            p.data = p.data.float()
    for n, b in list(mod._buffers.items()):
        if b is not None and b.dtype.is_floating_point and b.dtype not in FP8_DTYPES:
            mod._buffers[n] = b.float()
hf.eval()

# kernels = Counter(type(m).__name__ for m in hf.modules() if hasattr(m, "qweight"))
# print("Quantized layer classes actually in use:", dict(kernels))   # proves which kernel ran

# kernels = Counter(type(m).__name__ for m in hf.modules()
#                   if "Linear" in type(m).__name__ and type(m).__module__.startswith("gptqmodel"))
# print("Quantized layer classes actually in use:", dict(kernels))

kernels = Counter(type(m).__name__ for m in hf.modules()
    if "FP8" in type(m).__name__
)
print("FP8 layer classes actually in use:", dict(kernels))

dtype_mb = Counter()
for t in list(hf.parameters()) + list(hf.buffers()):
    dtype_mb[str(t.dtype)] += t.numel() * t.element_size() / MB
print("Storage by dtype (MB):", {k: round(v, 1) for k, v in dtype_mb.items()})


# ============================================================
# Run everything
# ============================================================

fid = fidelity(hf, check_ids, ref_logits)

run_benchmark(
    hf, tokenizer, DEVICE,
    label=f"gptq_{FP8_FORMAT}",
    model_name=MODEL_NAME,
    meta={
        "bits": 8, "group_size": None, 
        "scheme": "per-row scale, weight-only FP8, no calibration",
        "backend": str(INFER_BACKEND), "eager_dequant": EAGER_DEQUANT,
        "kernel_classes": dict(kernels), "load_time_s": load_time,
    },
    results_file=RESULTS_FILE,
    eval_source=EVAL_SOURCE,
    fp32_storage_bytes=fp32_bytes,
    num_params=num_params,
    fidelity_result=fid,
    ckpt_mb=checkpoint_size_mb(CHECKPOINT),
)

print("\nDone.")