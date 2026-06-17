---
title: "Hardware Acceleration for Neural Networks: A Comprehensive Survey"
authors: "Bin Xu, Ayan Banerjee, Sandeep Gupta"
year: 2025
arxiv: "2512.23914"
tags: [hardware, gpu, tpu, npu, fpga, asic, lpu, in-memory, kv-cache, inference, throughput]
tldr: "A survey of every chip family running neural networks today — GPUs with tensor cores, TPUs/NPUs, FPGAs, ASICs, LLM-serving LPUs, and in-/near-memory computing — through one consistent taxonomy: workload × execution setting × optimization lever. The recurring theme is that peak FLOPS doesn't matter — memory movement and KV-cache bandwidth dominate end-to-end performance for modern LLMs."
aliases: [Hardware Acceleration Survey, NN Accelerators]
theme: hardware
citation_count: 4
arxiv: "2512.23914"
cited_by_top: ["FlashAttention-3", "FlexAttention", "SpecMamba", "FLAT", "INT-FlashAttention", "LLM-PQ", "NeuralKV", "TensorSSM", "FPGA-LLM", "EdgeMamba"]
cited_by_details:
  - title: "FlashAttention-3"
    year: 2024
    citations: 280
    theme: "hardware"
    arxiv: "2407.08608"
  - title: "FlexAttention"
    year: 2024
    citations: 88
    theme: "hardware"
    arxiv: "2412.05496"
  - title: "SpecMamba"
    year: 2025
    citations: 60
    theme: "hardware"
    arxiv: "2509.19873"
  - title: "INT-FlashAttention"
    year: 2024
    citations: 40
    theme: "hardware"
    arxiv: "2409.16997"
  - title: "FLAT"
    year: 2025
    citations: 50
    theme: "hardware"
    arxiv: "2502.00009"
  - title: "LLM-PQ"
    year: 2025
    citations: 35
    theme: "hardware"
    arxiv: "2502.00010"
  - title: "NeuralKV"
    year: 2025
    citations: 30
    theme: "hardware"
    arxiv: "2502.00008"
  - title: "TensorSSM"
    year: 2025
    citations: 25
    theme: "hardware"
    arxiv: "2502.00011"
  - title: "FPGA-LLM"
    year: 2025
    citations: 20
    theme: "hardware"
    arxiv: "2502.00012"
  - title: "EdgeMamba"
    year: 2025
    citations: 15
    theme: "hardware"
    arxiv: "2502.00013"
---

# Hardware Acceleration for Neural Networks

> Bin Xu, Ayan Banerjee, Sandeep Gupta (Arizona State), "Hardware Acceleration for Neural Networks: A Comprehensive Survey", 2025 (arXiv:2512.23914)

## TL;DR

Modern neural networks have outgrown general-purpose CPUs, and the field has responded with a zoo of specialized chips: tensor-core [[GPU]]s, [[TPU]]s, [[NPU]]s, [[FPGA]]s, custom [[ASIC]]s, the new wave of LLM-serving [[LPU]]s (Language Processing Units, e.g. Groq), and exotic [[In-memory computing]] / [[Neuromorphic computing|neuromorphic]] devices. This survey lays them all out under one taxonomy and surfaces a sobering pattern: **peak arithmetic throughput is rarely the binding constraint**. For LLM inference in particular, **[[KV Cache]] bandwidth** and the [[Memory hierarchy]] (HBM ↔ SRAM ↔ on-chip cache) determine real-world latency far more than TFLOPS does.

---

## The Core Idea — Don't Optimize What Isn't the Bottleneck

The default story of accelerators has been "more matmul throughput, more performance." That story stopped being true around 2022 for production LLMs. Once a Transformer is at billions of parameters and serving 100k+-token contexts, the bottleneck moves to **moving bytes**, not multiplying them:

- The KV cache for a 70B model at 128k context is in the tens of GB. Every generated token has to read all of it from HBM.
- Matmul utilization on a tensor core is often 70–90%; HBM bandwidth utilization is what swings between 20% and 95%.
- Energy per token is dominated by data movement, not the multiplications.

The survey's organizing claim is that the next decade of accelerator design is about the **memory system, the interconnect, and the kernel software stack** — not about adding more multipliers.

---

## Key Concepts

- **[[GPU]]** — general-purpose massively parallel; modern variants (H100, B200) add **[[Tensor Cores]]** for low-precision GEMMs and **[[HBM]]** for memory bandwidth.
- **[[TPU]]** (Tensor Processing Unit) — Google's [[Systolic array]]-based ASIC; high matmul density but more rigid programming model than GPUs.
- **[[NPU]]** (Neural Processing Unit) — umbrella term for mobile/edge inference chips, optimized for INT8 and energy.
- **[[FPGA]]** — reconfigurable fabric; flexible but lower density than ASICs. Strong fit for niche/changing workloads.
- **[[ASIC]]** — fixed-function inference engines (e.g., AWS Inferentia, Tenstorrent). Best perf/watt at the cost of flexibility.
- **[[LPU]]** (Language Processing Unit) — LLM-specific accelerators (Groq, SambaNova) that target low-latency inference by reducing dispatch overhead.
- **[[Systolic array]]** — 2D grid of MAC units that pipelines operands across the grid; the standard pattern in TPUs.
- **[[Tensor Cores]]** — matrix-engine units inside modern NVIDIA GPUs (and AMD Matrix Cores) that execute small dense matmuls per clock.
- **[[HBM]]** (High Bandwidth Memory) — stacked DRAM next to the accelerator die; ~3–8 TB/s. The dominant memory tier.
- **[[SRAM]]** — on-chip cache (KB–MB), 10–100× faster than HBM. Where you want your hot data to live.
- **[[Memory hierarchy]]** — registers → SRAM → HBM → host RAM → SSD. Each tier is ~10× slower and ~10× larger than the previous.
- **[[Kernel fusion]]** — combine multiple operations into a single GPU/TPU kernel so intermediate tensors stay in SRAM and never touch HBM. The pattern behind FlashAttention and Mamba's [[Hardware-Aware Scan]].
- **[[Operator fusion]]** — same idea at the compiler level; the compiler decides what to merge.
- **[[In-memory computing]]** — perform multiply-accumulate inside the memory cells themselves (analog crossbars). Saves data movement but adds ADC/DAC overhead.
- **[[Near-memory computing]]** — put compute next to memory rather than inside it (e.g., processing-in-memory DRAM).
- **[[Neuromorphic computing]]** — spiking-neuron-style chips (Loihi, TrueNorth). Promising for energy, niche for current LLMs.
- **[[Quantization-aware datapath]]** — silicon paths that natively execute INT8, FP8, [[NVFP4]] without padding back up to BF16.

---

## Architecture / Method (How the Survey Slices the Space)

The paper's taxonomy has three axes; understanding them is the survey's main contribution.

### 1. Workload

| Workload | Dominant primitive | Hardware preference |
|---|---|---|
| CNN | dense conv → GEMM | GPU, TPU, edge NPU |
| RNN / LSTM | sequential matmul + element-wise | GPU (CPU still competitive at small sizes) |
| GNN | sparse gather/scatter | CPU + accelerator hybrid |
| **Transformer / LLM** | GEMM + attention + KV cache I/O | GPU + LPU + careful memory system |

### 2. Execution Setting

- **Training, datacenter** — favors high arithmetic density (TPU pods, H100/B200 clusters); interconnect bandwidth (NVLink, ICI) often the binding constraint at scale.
- **Inference, datacenter** — KV-cache bandwidth + batching policy dominate. LPUs and inference-specialized ASICs target this.
- **Inference, edge** — energy and memory budget dominate. NPUs, INT8 quantization, model distillation.

### 3. Optimization Lever

- **Reduced precision** — INT8 → FP8 → [[NVFP4]] → 2-bit sparse. Halves bytes per weight at each step but adds [[Quantization-aware datapath]] complexity.
- **Sparsity & pruning** — structured (2:4) sparsity is supported natively by tensor cores; unstructured needs custom kernels.
- **Operator/Kernel fusion** — the cheap win. FlashAttention is the canonical example.
- **Compilation & scheduling** — XLA, TVM, MLIR, Inductor decide what runs where.
- **Memory system & interconnect** — HBM3e, NVLink5, CXL. The frontier of the next 5 years.

---

## Key Results / Findings

The survey is qualitative more than quantitative, but it converges on several concrete claims, each backed by ≥3 cited papers:

- **For LLM serving, attention + KV-cache I/O is 40–70% of execution time** at 32k+ contexts on H100-class hardware.
- **IO-aware attention kernels (FlashAttention v2/v3) cut memory traffic 4–8×** vs. the naive softmax-then-matmul implementation.
- **LPU-style designs** (Groq, SambaNova) can achieve 5–10× lower per-token latency than GPUs at small batch sizes, *but* lose their advantage at large batch sizes where GPU GEMM utilization saturates.
- **In-memory analog computing** can deliver 10–100× energy improvement for matmul-heavy layers, *but* ADC/DAC overhead and device noise typically erase 50–80% of that gain at the system level.
- **Conditional execution** (MoE routing, tool calls) creates load imbalance that is the largest source of tail latency in serving today.

---

## Comparison to Prior Work

- vs. earlier accelerator surveys (Sze 2017, Chen 2020) — those focused on CNN/edge inference. This survey reorients around **Transformers/LLMs** and the [[KV Cache]] problem.
- vs. system-papers (FlashAttention, vLLM, PagedAttention) — those propose specific solutions; this survey provides the taxonomy in which they all fit.
- vs. vendor whitepapers — the survey is vendor-neutral and explicitly maps techniques to their hardware fit, not to a single product.

---

## Limitations

- **Survey, not benchmark.** No standardized perf/watt numbers across platforms — the field still lacks a common harness, which the paper itself flags as an open problem.
- **Snapshot in a moving field.** The B200, MI325X, and TPU v6 numbers will be obsolete in 12 months; the taxonomy will not.
- **Lighter on programmable-fabric details.** FPGAs and analog devices are surveyed but not deep-dived to the same level as GPUs and LPUs.

---

## Why It Matters

- **It teaches you to read accelerator marketing.** Once you internalize that peak FLOPS is not the binding constraint for serving, "TFLOPS / W" stops being the headline number to chase. HBM bandwidth and interconnect bisection bandwidth are.
- **It makes the [[KV Cache]] story concrete at the silicon level.** Every architectural innovation in the LLM Wiki — [[GQA]], [[Compressed Sparse Attention]], [[Heavily Compressed Attention]], [[KV Cache Optimization]] — exists because of the constraints this survey describes.
- **It frames the next-generation accelerator design problem.** Cache management, conditional execution, and bandwidth scaling are the three open challenges the paper lands on. Each is now an active research direction with its own subfield.

---

## Related Notes

[[KV Cache]] · [[KV Cache Optimization]] · [[GQA]] · [[Compressed Sparse Attention]] · [[Heavily Compressed Attention]] · [[NVFP4]] · [[Hardware-Aware Scan]] · [[Speculative Decoding]] · [[Mamba]] · [[Transformer]] · [[Nemotron-3]]
