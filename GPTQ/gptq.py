"""
GPTQ quantization with gptqmodel. Change BITS to make INT4 / INT8 checkpoints.
Run once per bit-width; the benchmark script then only loads the result.
"""

import os
import shutil
import time

import torch
from datasets import load_dataset
from gptqmodel import GPTQModel, GPTQConfig, BACKEND
from transformers import AutoTokenizer

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "EleutherAI/pythia-160m"

BITS = 2                      # vary bits everything remains same
GROUP_SIZE = 64               # 64 = same as your custom INT8 script (128 is gptqmodel's usual default)
DAMP_PERCENT = 0.1

OUTPUT_DIR = f"pythia-160m-gptq-int{BITS}"

# Quantization is a one-time job: use the GPU if you have one, otherwise the CPU is fine for 160M params.
QUANT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Calibration: GPTQ measures how weights react to REAL text, so more (and more varied) text = better.
NUM_CALIB_SAMPLES = 256
CALIB_SEQ_LEN = 512
SEED = 0
BATCH_SIZE = 1


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
ids = tokenizer("\n\n".join(train["text"]), return_tensors="pt")["input_ids"][0]

g = torch.Generator().manual_seed(SEED)
starts = torch.randint(0, ids.numel() - CALIB_SEQ_LEN - 1, (NUM_CALIB_SAMPLES,), generator=g)

calibration = []
for s in starts.tolist():
    window = ids[s:s + CALIB_SEQ_LEN]
    calibration.append({"input_ids": window, "attention_mask": torch.ones_like(window)})

print(f"Calibration: {len(calibration)} samples x {CALIB_SEQ_LEN} tokens")


# ============================================================
# Quantize
# ============================================================

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

quant_config = GPTQConfig(
    bits=BITS,
    group_size=GROUP_SIZE,
    sym=True,                 # symmetric, same as your custom script
    desc_act=True,           
    damp_percent=DAMP_PERCENT,
    act_group_aware=False,
    offload_to_disk=False,    # avoids the lm_head/offload save problem you hit before
)
print(quant_config)

model = GPTQModel.load(MODEL_NAME, quant_config, device=QUANT_DEVICE)

t0 = time.perf_counter()
model.quantize(
    calibration=calibration,
    batch_size=BATCH_SIZE,
    tokenizer=tokenizer,
    backend=BACKEND.GPTQ_TORCH,
)
print(f"Quantization time: {time.perf_counter() - t0:.1f} s")

model.save(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)   # the benchmark loads the tokenizer from here
print(f"Saved to {os.path.abspath(OUTPUT_DIR)}")