# FP16 benchmark for a Hugging Face causal LM.
#
# Same benchmark harness as FP32 / INT32:
# - same model
# - same tokenizer
# - same attention implementation
# - same latency methodology
# - same memory methodology
# - same perplexity text
#
# Only the model dtype changes from FP32 to FP16.

import gc
import json
import math
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "EleutherAI/pythia-160m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # ROCm also shows up as "cuda"
DTYPE = torch.float16
ATTN_IMPL = "sdpa"           # fastest exact attention; "eager" is the slow reference
SEED = 0

BATCH_SIZE = 1
NUM_WARMUP = 10
NUM_ITERATIONS = 50

PREFILL_LENGTHS = [32, 128, 512]   # prompt lengths for the prefill benchmark
DECODE_PROMPT_LEN = 32             # prompt length before token-by-token generation
DECODE_NEW_TOKENS = 64             # tokens generated per decode run
DECODE_RUNS = 5

MAX_EVAL_LEN = 1024                # window size for perplexity (Pythia supports 2048)
RESULTS_FILE = "benchmark_fp16.json"


# ============================================================
# Reproducibility / FP16 settings
# ============================================================

torch.manual_seed(SEED)
torch.set_float32_matmul_precision("highest")   # no TF32 / reduced-precision shortcuts
if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sync():
    torch.cuda.synchronize()


# ============================================================
# Model loading  (<-- the only part you change for other formats)
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
print("dtype:", next(model.parameters()).dtype, "| attention:", ATTN_IMPL)


# ============================================================
# Helpers
# ============================================================

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
    """Fixed random token ids: latency does not depend on token values, and this is reproducible."""
    g = torch.Generator().manual_seed(SEED)
    ids = torch.randint(0, len(tokenizer), (BATCH_SIZE, length), generator=g)
    return ids.to(DEVICE)


# ============================================================
# Model size
# ============================================================

# Count each tensor once (shared/tied weights would otherwise be counted twice).
seen, num_params, param_bytes = set(), 0, 0
for p in model.parameters():
    key = (p.data_ptr(), p.numel())
    if key in seen:
        continue
    seen.add(key)
    num_params += p.numel()
    param_bytes += p.numel() * p.element_size()
buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
total_bytes = param_bytes + buffer_bytes
MB = 1024 ** 2

print("\n--- MODEL SIZE ---")
print(f"Parameters:        {num_params:,}")
print(f"Parameter memory:  {param_bytes / MB:.2f} MB")
print(f"Buffer memory:     {buffer_bytes / MB:.2f} MB")
print(f"Total storage:     {total_bytes / MB:.2f} MB")
print(f"Bits per param:    {8 * total_bytes / num_params:.2f}")


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
print(f"Activation memory: {(peak - baseline) / MB:.2f} MB")

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

# Non-overlapping windows. Each window predicts (len - 1) tokens because the
# first token has nothing before it. We weight each window by its predicted-token count.
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

print("\n--- SUMMARY ---")
print(f"Parameters:        {num_params:,}")
print(f"Model storage:     {total_bytes / MB:.2f} MB")
for L, s in prefill_results.items():
    print(f"Prefill len={L:<4}  {s['median_ms']:.3f} ms  ({s['tokens_per_sec']:.1f} tok/s)")
print(f"Decode:            {decode_stats['median_ms']:.3f} ms/token  ({decode_stats['tokens_per_sec']:.1f} tok/s)")
print(f"Loss / PPL:        {avg_loss:.6f} / {perplexity:.6f}")

results = {
    "env": {
        "model": MODEL_NAME,
        "dtype": str(DTYPE),
        "attention": ATTN_IMPL,
        "device": DEVICE,
        "device_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "batch_size": BATCH_SIZE,
        "warmup": NUM_WARMUP,
        "iterations": NUM_ITERATIONS,
    },
    "size": {"params": num_params, "storage_mb": total_bytes / MB},
    "prefill": {str(k): v for k, v in prefill_results.items()},
    "decode": decode_stats,
    "memory": memory_results,
    "perplexity": {
        "dataset": eval_name,
        "tokens_predicted": total_pred,
        "loss": avg_loss,
        "ppl": perplexity,
    },
}
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {RESULTS_FILE}")