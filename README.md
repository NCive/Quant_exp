# Quant_exp: Comprehensive LLM Quantization & Benchmark Suite

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](file:///d:/Python%20Projects/Quant_exp/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x%20%7C%20ROCm%20%7C%20CUDA-orange.svg)](https://pytorch.org)
[![HuggingFace Transformers](https://img.shields.io/badge/🤗%20Transformers-Pythia--160M-yellow.svg)](https://huggingface.co/EleutherAI/pythia-160m)

An empirical research and benchmarking testbed exploring the frontiers of **Large Language Model (LLM) quantization**. This repository provides complete, standalone implementations and controlled micro-benchmarks comparing traditional integer quantization, modern sub-byte floating-point standards (**FP8**, **FP4**, **MXFP4**, **NVFP4**), novel geometric vector quantizers (**TurboQuant**, **PolarQuant**), and Hessian-guided second-order post-training quantization (**GPTQ** from 2 to 8 bits).

All experiments are benchmarked under a unified evaluation harness using [`EleutherAI/pythia-160m`](https://huggingface.co/EleutherAI/pythia-160m) under identical prompt distributions, sequence lengths, attention backends (`sdpa`), and device synchronizations.

> **Important — Custom Quantization Runtime:** The custom quantization formats in
> `Custom_Made/` are software/fake-quantized implementations. Quantized weights
> are unpacked and dequantized to FP32 during inference. Therefore, their latency
> measurements primarily characterize the implemented software pipeline rather
> than the theoretical throughput of dedicated FP4/FP8/INT4 hardware. This is
> particularly relevant when interpreting results on the AMD Radeon 780M.

---

## Table of Contents

- [Key Highlights](#key-highlights)
- [Quantization Paradigms Implemented](#quantization-paradigms-implemented)
  - [1. Standard Baselines (FP32, FP16, BF16)](#1-standard-baselines-fp32-fp16-bf16)
  - [2. Uniform Integer Quantization (INT32, INT16, INT8, INT4)](#2-uniform-integer-quantization-int32-int16-int8-int4)
  - [3. Sub-Byte & Microscaling Floating-Point Formats](#3-sub-byte--microscaling-floating-point-formats)
  - [4. Geometric & Codebook Vector Quantization](#4-geometric--codebook-vector-quantization)
  - [5. Second-Order Post-Training Quantization (GPTQ)](#5-second-order-post-training-quantization-gptq)
- [Empirical Benchmark Results](#empirical-benchmark-results)
  - [Custom Implementation Benchmark Table](#custom-implementation-benchmark-table)
  - [GPTQ Bitwidth Ladder Table](#gptq-bitwidth-ladder-table)
- [In-Depth Technical Insights](#in-depth-technical-insights)
  - [The Perplexity Cliff: Naive INT4 vs. GPTQ INT4](#the-perplexity-cliff-naive-int4-vs-gptq-int4)
  - [FP8 Precision vs. Dynamic Range: E4M3FN vs. E5M2](#fp8-precision-vs-dynamic-range-e4m3fn-vs-e5m2)
  - [Microscaling Face-Off: NVFP4 vs. MXFP4 vs. FP4](#microscaling-face-off-nvfp4-vs-mxfp4-vs-fp4)
  - [Geometric Quantization: TurboQuant vs. PolarQuant](#geometric-quantization-turboquant-vs-polarquant)
  - [Memory Accounting: Quantized Layer vs. Global Model Footprint](#memory-accounting-quantized-layer-vs-global-model-footprint)
- [Repository Structure](#repository-structure)
- [Quickstart & Reproduction Guide](#quickstart--reproduction-guide)
  - [Prerequisites](#prerequisites)
  - [Running Custom Scratch Benchmarks](#running-custom-scratch-benchmarks)
  - [Running GPTQ Quantization & Evaluation](#running-gptq-quantization--evaluation)
- [Benchmarking Protocol & Hardware](#benchmarking-protocol--hardware)
- [License & Author](#license--author)

---

## Key Highlights

- **Dual-Track Architecture**:
  - **Custom Scratch Implementations ([`Custom_Made/`](file:///d:/Python%20Projects/Quant_exp/Custom_Made))**: Standalone mathematical simulations of quantization schemes with custom bit-packing, dequantization routines, and layer replacement wrappers.
  - **Hessian-Guided PTQ ([`GPTQ/`](file:///d:/Python%20Projects/Quant_exp/GPTQ))**: Real second-order error minimization using [`gptqmodel`](https://github.com/ModelCloud/GPTQModel) across INT2 through INT8, along with data-free FP8 conversions.
- **Microscaling Formats Evaluated**:
  - **OCP MXFP4**: 32-element blocks with 8-bit unsigned power-of-two scale ($E8M0$, scale factor $2^{s-127}$).
  - **NVIDIA Blackwell NVFP4**: 16-element blocks with dual-level scaling (per-block $FP8\text{ }E4M3$ scale + per-tensor $FP32$ scale).
- **Novel Geometric Formats**:
  - **TurboQuant**: Randomized Walsh-Hadamard transform for coordinate decorrelation + Lloyd-Max scalar codebook fitted to Beta-distributed marginals on the unit sphere.
  - **PolarQuant**: Parameter-free hyperspherical angle decomposition with uniform polar grid quantization.
- **Rigorously Controlled Harness**:
  - Identical prefill sequences ($32$, $128$, $512$ tokens) and decode sweeps ($32$ prompt tokens $\rightarrow$ $64$ generated tokens).
  - Explicit GPU synchronization (`torch.cuda.synchronize()`), garbage-collector pausing during timed regions, and exact perplexity evaluation on held-out text.

---

## Quantization Paradigms Implemented

### 1. Standard Baselines (FP32, FP16, BF16)
- [`FP32.py`](file:///d:/Python%20Projects/Quant_exp/Custom_Made/FP32.py): Full single-precision reference running with highest matmul precision (no TF32) and Flash-style SDPA attention.
- [`FP16.py`](file:///d:/Python%20Projects/Quant_exp/Custom_Made/FP16.py): Half-precision float (5 exponent bits, 10 mantissa bits), serving as the standard memory/throughput baseline.
- [`BF16.py`](file:///d:/Python%20Projects/Quant_exp/Custom_Made/BF16.py): Bfloat16 (8 exponent bits, 7 mantissa bits), matching FP32's dynamic range at reduced precision.

### 2. Uniform Integer Quantization (INT32, INT16, INT8, INT4)
- **Symmetric Group-Wise Linear Quantization**:
  $$\text{scale} = \max\left(\frac{\max(|W|)}{q_{\max}}, \epsilon\right), \quad Q = \text{clamp}\left(\left\lfloor \frac{W}{\text{scale}} \right\rceil, -q_{\max}, q_{\max}\right)$$
  where weights are partitioned into groups along `in_features` ($G=64$).
- [`Int8.py`](file:///d:/Python%20Projects/Quant_exp/Custom_Made/Int8.py): Maps weights to $[-127, +127]$.
- [`Int4.py`](file:///d:/Python%20Projects/Quant_exp/Custom_Made/Int4.py): Maps weights to $[-7, +7]$ with bit packing (storing two 4-bit two's complement codes into a single `uint8` buffer).
- [`Int16.py`](file:///d:/Python%20Projects/Quant_exp/Custom_Made/Int16.py) and [`Int32.py`](file:///d:/Python%20Projects/Quant_exp/Custom_Made/Int32.py): High-bit integer validation scripts verifying quantization behavior and bounds.

### 3. Sub-Byte & Microscaling Floating-Point Formats
- [`FP8_e4m3fn.py`](file:///d:/Python%20Projects/Quant_exp/Custom_Made/FP8_e4m3fn.py): 8-bit float with 4 exponent bits and 3 mantissa bits (maximum representable value $448$, no infinity). Offers superior precision for weight representation.
- [`FP8_e5m2.py`](file:///d:/Python%20Projects/Quant_exp/Custom_Made/FP8_e5m2.py): 8-bit float with 5 exponent bits and 2 mantissa bits (maximum value $57344$). Offers wide dynamic range at the expense of precision.
- [`FP4_e2m1.py`](file:///d:/Python%20Projects/Quant_exp/Custom_Made/FP4_e2m1.py): Microscopic 4-bit floating point format with values $\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$ and symmetric negative counterparts, packed two per byte with an $FP32$ scale per group of 64.
- [`MXFP4.py`](file:///d:/Python%20Projects/Quant_exp/Custom_Made/MXFP4.py): **OCP Microscaling Formats standard**.
  - Block size: 32 weights share one scale.
  - Scale format: $E8M0$ (unsigned 8-bit power-of-two exponent, $\text{scale} = 2^{s-127}$).
  - Storage cost: $4\text{ bits} + \frac{8\text{ bits}}{32} = 4.25\text{ bits/weight}$.
- [`NVFP4.py`](file:///d:/Python%20Projects/Quant_exp/Custom_Made/NVFP4.py): **NVIDIA Blackwell FP4 microscaling specification**.
  - Block size: 16 weights share one scale.
  - Two-level scale: Each block stores an $FP8\text{ }E4M3$ scale, multiplied by a per-layer global $FP32$ tensor scale.
  - Storage cost: $4\text{ bits} + \frac{8\text{ bits}}{16} + \mathcal{O}(1) \approx 4.5\text{ bits/weight}$.

### 4. Geometric & Codebook Vector Quantization
- [`TurboQuant.py`](file:///d:/Python%20Projects/Quant_exp/Custom_Made/TurboQuant.py):
  1. Partitions weights into blocks of size $D=128$.
  2. Extracts vector norm $r = \|w\|_2$ (stored in $FP16$).
  3. Projects direction $u = w / r$ onto an orthonormal basis via randomized **Walsh-Hadamard transform** ($H_D$) to decorrelate coordinates.
  4. Encodes rotated coordinates using an optimal 16-point **Lloyd-Max codebook** fitted to the theoretical $\text{Beta}\left(\frac{1}{2}, \frac{D-1}{2}\right)$ marginal distribution of points uniformly distributed on $\mathbb{S}^{D-1}$.
  5. Packs indices into 4 bits (2 codes/byte).
- [`Polar_Quant.py`](file:///d:/Python%20Projects/Quant_exp/Custom_Made/Polar_Quant.py):
  - Converts $D$-dimensional blocks into hyperspherical coordinates: radius $r$ and $D-1$ angles $(\phi_1, \dots, \phi_{D-2} \in [0, \pi], \phi_{D-1} \in [-\pi, \pi])$.
  - Quantizes each angle uniformly across 16 bins ($4\text{ bits}$), reconstructing weights on the forward pass via trigonometric cumulative products without requiring data-dependent codebook optimization.

### 5. Second-Order Post-Training Quantization (GPTQ)
- [`gptq.py`](file:///d:/Python%20Projects/Quant_exp/GPTQ/gptq.py) & [`benchmark_gptq.py`](file:///d:/Python%20Projects/Quant_exp/GPTQ/benchmark_gptq.py):
  - Employs the Generalized Post-Training Quantization algorithm using inverse Hessian matrices:
    $$w_q^* = \arg\min_{w_q} (w - w_q)^T H (w - w_q)$$
  - Calibration: 256 samples of 512 tokens from Wikitext-2.
  - Configuration: Group size $64$, symmetric, `desc_act=True` (activation ordering), damping factor $10\%$.
  - Bitwidth ladder: **INT2, INT3, INT4, INT5, INT6, INT7, INT8**.
- [`gptq_model_fp8.py`](file:///d:/Python%20Projects/Quant_exp/GPTQ/gptq_model_fp8.py): Data-free row-wise conversion to $FP8\text{ }E4M3FN$ and $FP8\text{ }E5M2$.

---

## Empirical Benchmark Results

All benchmarks were run on an AMD Ryzen 7 8845HS / AMD Radeon 780M platform using PyTorch with ROCm/HIP. Batch size $= 1$, deterministic seed $= 0$, SDPA attention.

### Custom Implementation Benchmark Table

The table below summarizes results from [`Custom_Made/`](file:///d:/Python%20Projects/Quant_exp/Custom_Made) across all formats, sorted from baseline precision down to sub-byte formats.

| Format | Format Class | Effective Bits/Param | Model Storage (MB) | Weight Relative RMSE | Prefill 512 (tok/s) | Decode (tok/s) | Eval Loss | Perplexity (PPL) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 Baseline** | Float32 | 32.00 | 619.2 MB | Baseline | 1,763.7 | 36.1 | 2.9439 | **18.99** |
| **FP16** | Float16 | 16.00 | 309.6 MB | Minimal | **7,995.9** | **111.2** | 2.9500 | **19.11** |
| **BF16** | Bfloat16 | 16.00 | 309.6 MB | Minimal | **8,001.7** | 89.4 | 3.4255 | 30.74 |
| **INT32** | Uniform Int | 32.38 | 626.6 MB | < 0.0001 | 1,927.9 | 27.0 | 2.9439 | **18.99** |
| **INT16** | Uniform Int | 20.20 | 390.9 MB | < 0.0001 | 1,911.9 | 29.1 | 2.9439 | **18.99** |
| **INT8 (g64)** | Uniform Int | 14.11 | 273.0 MB | 0.0031 | 1,974.3 | 29.7 | 2.9564 | **19.23** |
| **FP8 E4M3FN** | Sub-byte Float | 14.11 | 273.0 MB | 0.0262 | 1,819.0 | 23.4 | 3.1990 | **24.51** |
| **FP8 E5M2** | Sub-byte Float | 14.11 | 273.0 MB | 0.0525 | 1,362.0 | 19.3 | 3.9506 | 51.97 |
| **INT4 (g64)** | Uniform Int | 11.07 | 214.1 MB | 0.0763 | 1,724.3 | 19.0 | 5.0327 | 153.35 |
| **TurboQuant** | Vector Lloyd-Max | 10.78 | 208.7 MB | 0.0948 | 1,042.4 | 6.2 | 9.1706 | 9,610.72 |
| **NVFP4** | Blackwell Microscale | 11.07 | 214.1 MB | 0.1030 | 1,582.6 | 17.5 | 6.9117 | 1,003.91 |
| **MXFP4** | OCP Microscale | 10.88 | 210.4 MB | 0.1095 | 1,841.1 | 19.5 | 10.7146 | 45,010.24 |
| **FP4 E2M1 (g64)**| Sub-byte Float | 11.07 | 214.1 MB | 0.1106 | 1,884.6 | 20.6 | 11.0013 | 59,953.39 |
| **PolarQuant** | Hyperspherical Grid | 10.78 | 208.6 MB | 0.3585 | 1,147.3 | 4.5 | 22.4370 | 5.55 × 10⁹ |

> *Note on latency*: All custom formats operate in "fake quantization" mode (unpacking/dequantizing on-the-fly to FP32 during forward passes). As such, throughput reflects the computational overhead of the software dequantization kernel on CPU/iGPU rather than dedicated tensor hardware acceleration.

---

### GPTQ Bitwidth Ladder Table

The table below summarizes results from [`GPTQ/`](file:///d:/Python%20Projects/Quant_exp/GPTQ) running on the same evaluation text, showing how Hessian-guided compensation pushes low-bit accuracy far beyond naive round-to-nearest methods.

| Configuration | Bitwidth | Group Size | Calibration | Checkpoint Size | Peak Memory | Decode (tok/s) | Eval Loss | Perplexity (PPL) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 Unquantized**| 32-bit | N/A | None | 619.2 MB | 752.7 MB | 36.1 | 2.9439 | **18.99** |
| **GPTQ INT8** | 8-bit | 64 | WikiText-2 | 382.8 MB | 518.6 MB | 17.1 | 2.9413 | **18.94** *(+0.05 vs FP32)* |
| **GPTQ INT7** | 7-bit | 64 | WikiText-2 | 372.5 MB | 506.5 MB | 3.2 | 2.9441 | **18.99** |
| **GPTQ INT6** | 6-bit | 64 | WikiText-2 | 362.2 MB | 496.5 MB | 4.1 | 2.9621 | **19.34** |
| **GPTQ INT5** | 5-bit | 64 | WikiText-2 | 351.9 MB | 486.8 MB | 4.6 | 2.9881 | **19.85** |
| **GPTQ INT4** | 4-bit | 64 | WikiText-2 | 341.7 MB | 477.1 MB | 12.5 | 3.1139 | **22.51** |
| **GPTQ INT3** | 3-bit | 64 | WikiText-2 | 331.4 MB | 464.4 MB | 9.4 | 3.6617 | **38.93** |
| **GPTQ INT2** | 2-bit | 64 | WikiText-2 | 321.1 MB | 455.6 MB | 14.9 | 7.0122 | 1,110.10 |
| **GPTQ-FP8 E4M3FN**| 8-bit | Row | Data-free | 619.5 MB | 754.2 MB | 12.5 | 2.9476 | **19.06** |
| **GPTQ-FP8 E5M2** | 8-bit | Row | Data-free | 619.5 MB | 754.2 MB | 13.0 | 26.7129 | 3.99 × 10¹¹ *(Diverged)* |

---

## In-Depth Technical Insights

### The Perplexity Cliff: Naive INT4 vs. GPTQ INT4
When quantizing to 4 bits without calibration data, round-to-nearest uniform quantization ([`Custom_Made/Int4.py`](file:///d:/Python%20Projects/Quant_exp/Custom_Made/Int4.py)) degrades perplexity from **18.99 to 153.35** because correlated weight errors accumulate across transformer layers. In contrast, GPTQ with second-order Hessian error compensation ([`GPTQ/gptq.py`](file:///d:/Python%20Projects/Quant_exp/GPTQ/gptq.py)) brings 4-bit perplexity down to **22.51**—preserving language modeling capability within $3.5$ PPL points of unquantized FP32.

```
PPL Comparison @ 4-Bit:
  FP32 Baseline :  18.99  ███
  GPTQ INT4     :  22.51  ████
  Naive INT4    : 153.35  ██████████████████████████
```

### FP8 Precision vs. Dynamic Range: E4M3FN vs. E5M2
Both $FP8$ variants consume identical storage ($8\text{ bits}$), but allocate bits differently:
- **E4M3FN** (4 exponent, 3 mantissa) dedicates higher resolution to weight distribution within a narrower range. It achieves **24.51 PPL** in custom group-quantization and **19.06 PPL** in GPTQModel.
- **E5M2** (5 exponent, 2 mantissa) provides huge dynamic range (up to 57,344) but only 4 discrete positive values per octave. As a result, quantization noise destroys weight fidelity, yielding **51.97 PPL** in group-wise mode and causing complete divergence in uncalibrated row-wise conversion.
- **Conclusion**: For LLM weight quantization without outlier protection, **E4M3FN is decisively superior to E5M2**.

### Microscaling Face-Off: NVFP4 vs. MXFP4 vs. FP4
At 4-bit representations, Microscaling formats address weight outliers by shrinking the scaling granularity:
1. **Generic FP4 (Group 64, FP32 scale)**: PPL = **59,953**. Fixed grid spacing cannot accommodate heavy-tailed distributions.
2. **OCP MXFP4 (Block 32, E8M0 scale)**: PPL = **45,010**. While block size is smaller ($32$), constraining the scale factor to strict powers of two ($2^{s-127}$) causes severe rounding errors.
3. **NVIDIA NVFP4 (Block 16, FP8 E4M3 block scale + FP32 tensor scale)**: PPL = **1,003.91**. By halving the block size to 16 and using continuous $FP8$ scale values, NVFP4 reduces relative reconstruction RMSE to $0.1030$ and cuts perplexity by over $40\times$ compared to MXFP4.

### Geometric Quantization: TurboQuant vs. PolarQuant
- **TurboQuant** rotates weight vectors with a randomized Walsh-Hadamard transform, converting arbitrary directional dependencies into independent marginals that conform to a Beta distribution. By quantizing along this known distribution with an optimal 16-point Lloyd-Max codebook, it achieves **RMSE = 0.0948** and **PPL = 9,610**.
- **PolarQuant** uses a closed-form hyperspherical transformation into angles $\phi_1, \dots, \phi_{D-1}$. However, high-dimensional spheres ($D=128$) concentrate mass near the equator ($\phi \approx \pi/2$). A uniform angle grid wastes code words on sparsely populated polar regions, causing angular errors to cascade through cumulative sine products during dequantization (**PPL = 5.55 × 10⁹**).
- **Takeaway**: Rotation + optimal marginal codebooks (TurboQuant) vastly outperform raw uniform coordinate systems (PolarQuant) on hyperspheres.

### Memory Accounting: Quantized Layer vs. Global Model Footprint
Why does an INT4 model report ~11 bits/parameter rather than 4 bits/parameter?
In [`EleutherAI/pythia-160m`](https://huggingface.co/EleutherAI/pythia-160m):
- Total parameters: **162,322,944**
- Quantized Linear parameters: **123,568,128** ($76.1\%$ of total)
- Unquantized parameters: Token embeddings (`embed_in`) and language model head (`embed_out`) retain 32-bit representations ($38,754,816$ parameters $\approx 147.8\text{ MB}$).
- Combined with layer scale buffers, overall storage drops from **619.2 MB to 214.1 MB**, yielding an effective whole-model footprint of **11.07 bits/param**.

---

## Repository Structure

```text
Quant_exp/
├── Custom_Made/                     # Scratch custom quantization implementations & harnesses
│   ├── FP32.py                      # FP32 reference baseline (SDPA, seed 0, highest precision)
│   ├── FP16.py                      # Half-precision FP16 baseline
│   ├── BF16.py                      # Brain Floating Point BF16 baseline
│   ├── Int32.py                     # 32-bit integer simulation
│   ├── Int16.py                     # 16-bit integer simulation
│   ├── Int8.py                      # 8-bit group-wise symmetric integer quantization
│   ├── Int4.py                      # 4-bit group-wise integer quantization (packed uint8)
│   ├── FP8_e4m3fn.py                # 8-bit float (4 exponent, 3 mantissa)
│   ├── FP8_e5m2.py                  # 8-bit float (5 exponent, 2 mantissa)
│   ├── FP4_e2m1.py                  # 4-bit float (2 exponent, 1 mantissa, packed)
│   ├── MXFP4.py                     # OCP Microscaling Formats standard (block 32, E8M0 scale)
│   ├── NVFP4.py                     # NVIDIA Blackwell FP4 format (block 16, FP8 E4M3 scale)
│   ├── TurboQuant.py                # Walsh-Hadamard rotation + Lloyd-Max codebook vector quant
│   ├── Polar_Quant.py               # Hyperspherical polar coordinate angle quantizer
│   └── benchmark_*.json             # Raw benchmark output logs and metrics for each format
├── GPTQ/                            # Second-order Post-Training Quantization suite
│   ├── gptq.py                      # GPTQ calibration & quantization pipeline (INT2 - INT8)
│   ├── gptq_model_fp8.py            # Data-free FP8 quantization (E4M3FN & E5M2)
│   ├── benchmark_gptq.py            # Standardized benchmark runner for GPTQ checkpoints
│   ├── pythia-160m-gptq-int[2-8]/   # Saved GPTQ checkpoints per bitwidth
│   ├── pythia-160m-gptq-fp8-*/      # Saved GPTQ FP8 checkpoints
│   └── benchmark_gptq_*.json        # Benchmark output records for GPTQ checkpoints
├── LICENSE                          # MIT License
└── README.md                        # Documentation & analysis
```

---

## Quickstart & Reproduction Guide

### Prerequisites
Ensure Python 3.10+ and a PyTorch installation matching your hardware (CUDA or ROCm) are installed:

```bash
# Clone the repository
git clone https://github.com/NCive/Quant_exp.git
cd Quant_exp

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux:
source venv/bin/activate

# Install dependencies
pip install torch transformers datasets psutil gptqmodel
```

### Running Custom Scratch Benchmarks
To benchmark any custom quantization method, simply run the corresponding script in [`Custom_Made/`](file:///d:/Python%20Projects/Quant_exp/Custom_Made):

```bash
# Run INT8 benchmark
python Custom_Made/Int8.py

# Run NVIDIA NVFP4 microscaling benchmark
python Custom_Made/NVFP4.py

# Run TurboQuant geometric vector quantization benchmark
python Custom_Made/Turbo_Quant.py
```
Each run prints detailed layer counts, storage compression ratios, prefill throughput, decode throughput, and perplexity, and writes a structured JSON report to `benchmark_<format>.json`.

### Running GPTQ Quantization & Evaluation

#### 1. Quantize a model to a target bitwidth
Edit `BITS` in [`GPTQ/gptq.py`](file:///d:/Python%20Projects/Quant_exp/GPTQ/gptq.py) (e.g., `BITS = 4`), then run:
```bash
python GPTQ/gptq.py
```
This calibrates on Wikitext-2 and exports a safetensors checkpoint to `pythia-160m-gptq-int<BITS>/`.

#### 2. Run the GPTQ Benchmark
Specify the target checkpoint in [`GPTQ/benchmark_gptq.py`](file:///d:/Python%20Projects/Quant_exp/GPTQ/benchmark_gptq.py) and execute:
```bash
python GPTQ/benchmark_gptq.py
```

---

## Benchmarking Protocol & Hardware

To guarantee strict reproducibility across all runs:
1. **Attention Backend:** The attention implementation is fixed within each benchmark family; custom benchmarks use SDPA where configured, while the GPTQ evaluation uses eager attention.
2. **Warmup & Iteration**: 10 warmup runs followed by 50 timed iterations per test length.
3. **Garbage Collection Isolation**: `gc.disable()` is engaged during timing loops to prevent GC pauses from contaminating kernel latency measurements.
4. **Device Synchronization**: `torch.cuda.synchronize()` is enforced before and after all time-stamped operations.
5. **Evaluation Set**: Built-in fixed prompt text (800 tokens, 799 prediction steps) evaluated with batch size 1 under deterministic seeds.
6. **Execution Platform**:
   - **Processor**: AMD Ryzen 7 8845HS (8 physical cores, 16 threads)
   - **Graphics / Accelerator**: AMD Radeon 780M Graphics (ROCm / HIP)
   - **OS**: Windows / Linux compatible

---

## License & Author

Developed by **Neil Dhere** ([@NCive](https://github.com/NCive)).

Released under the [MIT License](file:///d:/Python%20Projects/Quant_exp/LICENSE).
