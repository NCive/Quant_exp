"""
FP8 quantization with gptqmodel (FP8Config).

Note: this is NOT the GPTQ algorithm. FP8 in gptqmodel is a data-free conversion:
each weight row gets a scale, then values are cast to FP8. No calibration data,
no Hessian, no error compensation. So there is no group size and no desc_act here.
"""

import glob
import os
import shutil
import time

import torch
from gptqmodel import GPTQModel, BACKEND
from gptqmodel.quantization import FP8Config
from transformers import AutoTokenizer

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "EleutherAI/pythia-160m"

FP8_FORMAT = "float8_e5m2"        # "float8_e4m3fn" or "float8_e5m2" 
OUTPUT_DIR = f"pythia-160m-gptq-fp8-{FP8_FORMAT.split('_')[1]}"   # -> ...-fp8-e4m3fn

QUANT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ============================================================
# Quantize
# ============================================================

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

quant_config = FP8Config(
    format=FP8_FORMAT,
    bits=8,
    weight_scale_method="row",      # one scale per output row (how gptqmodel's FP8 checkpoints are made)
    offload_to_disk=False,          # FP8Config turns this ON by default; it crashed save() on lm_head
                                    # (a 160M model fits in RAM easily, same fix as your GPTQ script)
)
print(quant_config)

model = GPTQModel.load(MODEL_NAME, quant_config, device=QUANT_DEVICE)

t0 = time.perf_counter()
model.quantize(
    calibration=None,               # FP8 needs no calibration data
    backend=BACKEND.FP8_TORCH,
)
print(f"Quantization time: {time.perf_counter() - t0:.1f} s")

model.save(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)


# ============================================================
# Check that weights were really written
# ============================================================

files = glob.glob(os.path.join(OUTPUT_DIR, "*.safetensors"))
assert files, f"No .safetensors weight files saved in {OUTPUT_DIR}!"
for f in files:
    print(f"  {os.path.basename(f)}: {os.path.getsize(f) / 1024**2:.1f} MB")

print(f"Saved to {os.path.abspath(OUTPUT_DIR)}")