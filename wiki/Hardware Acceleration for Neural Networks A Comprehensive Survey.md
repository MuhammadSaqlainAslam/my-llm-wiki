---
title: Hardware Acceleration for Neural Networks: A Comprehensive Survey
authors: Bin Xu, Ayan Banerjee, Sandeep Gupta
year: 2026
arxiv: 2512.23914
tags: [hardware, inference, attention, foundational, benchmarks, moe]
citation_count: 0
tldr: A comprehensive survey of neural network hardware acceleration covering GPUs, TPUs, FPGAs, ASICs, LPUs, and in-memory computing, showing that data movement rather than peak arithmetic throughput is the dominant bottleneck across all modern accelerator classes.
---

## The Problem

Modern neural networks have grown explosively in size, diversity, and deployment context — from billion-parameter LLMs in datacenters to always-on classifiers on battery-powered edge devices. The naive assumption that "more FLOPs = faster models" breaks down quickly in practice. Peak arithmetic throughput is almost never the real bottleneck; instead, it's the cost of *moving data* — across DRAM, caches, and network interconnects — that dominates end-to-end runtime and energy. For a concrete reference point: reading a single byte from off-chip DRAM costs orders of magnitude more energy than performing a floating-point multiply-accumulate.

This problem is getting worse, not better. Transformers and LLMs introduce the KV-cache, whose memory footprint scales with both sequence length and batch size, creating bandwidth demands that outpace arithmetic capacity by a wide margin. Mixture-of-Experts (MoE) routing, dynamic sparsity, and irregular control flow further break the assumptions that classical GPU designs were built on — namely, dense, regular, highly predictable matrix multiplications.

The research community has responded by building a fragmented zoo of accelerators: GPUs with tensor cores, Google's TPUs, mobile NPUs, FPGA overlays, ASIC inference engines, and specialized LLM-serving chips (LPUs). But there has been no unified framework for reasoning about what each of these does well, where it fails, and how the software stack (compilers, runtimes, kernel libraries) determines whether hardware potential is ever realized. That gap is what this survey fills.

## The Idea

The core organizing insight is that hardware acceleration for neural networks is a *three-axis co-design problem*: workload type (CNN, RNN, GNN, Transformer/LLM), execution setting (training vs. inference, datacenter vs. edge), and optimization lever (precision, sparsity, operator fusion, memory hierarchy design). No single axis can be optimized in isolation — pulling on one changes the bottleneck on another.

Think of it like a pipeline with three valves: compute, memory bandwidth, and communication. Closing the compute valve (via quantization) just shifts the bottleneck to memory. Opening the sparsity valve can improve utilization but causes load-imbalance headaches. The survey's contribution is a unified taxonomy that lets you reason about all three valves simultaneously, rather than optimizing them one at a time.

## How It Works

**Taxonomy of hardware platforms:**
- **GPUs + tensor cores**: General-purpose, massively parallel, excellent for dense matrix math. Tensor cores perform mixed-precision matrix multiply-accumulate (e.g., FP16 multiply, FP32 accumulate) in a single cycle on small tiles (e.g., 16×16 in NVIDIA's architecture). High-Bandwidth Memory (HBM) partially alleviates the memory wall.
- **TPUs / NPUs**: Domain-specific chips built around systolic arrays — a grid of processing elements where data flows in a wave, each cell computing a partial product and passing results to its neighbor. This eliminates the memory re-fetch overhead of conventional caches for dense matrix ops.
- **FPGAs**: Reconfigurable logic that can be tailored exactly to a model's operator mix. Useful when the workload is stable and the precision/sparsity pattern is known ahead of time; poor at adapting to dynamic workloads.
- **ASICs**: Fixed-function inference engines, maximally efficient for their target workload (e.g., a specific quantized CNN), but expensive to redesign. Best for high-volume, stable deployments.
- **LPUs (Language Processing Units)**: Emerging class of chip targeting predictable, low-latency autoregressive token generation — the key bottleneck in LLM serving where memory bandwidth (not compute) is the binding constraint.
- **In-/near-memory computing**: Rather than shipping data to the compute unit, perform arithmetic inside or adjacent to the memory array. Dramatically reduces data movement energy, but programming models and precision are challenging.
- **Neuromorphic / analog**: Event-driven, spike-based execution for ultra-sparse, ultra-low-power workloads. Promising but not yet mainstream for standard DNN inference.

**Key optimization levers:**

*Reduced precision*: Going from FP32 → FP16 → INT8 → INT4 cuts both memory footprint and arithmetic energy, often with minimal accuracy loss when combined with quantization-aware training. The critical engineering detail is keeping accumulators in higher precision (e.g., FP32) even when inputs are FP16, to preserve numerical stability.

*Sparsity and pruning*: Zero weights can be skipped, halving compute and bandwidth — in theory. In practice, unstructured sparsity is hard to exploit efficiently because irregular memory access patterns break the assumptions of SIMD pipelines. Structured sparsity (e.g., NVIDIA's 2:4 sparsity format, where 2 of every 4 weights are zero) is hardware-friendly and achieves ~2× speedup with modest accuracy cost.

*Operator fusion*: Instead of writing intermediate activations back to DRAM between kernels (e.g., matmul → layernorm → activation), fuse them into a single kernel that keeps data in on-chip SRAM. FlashAttention is the canonical example: it recomputes rather than stores the attention matrix, reducing memory complexity from O(N²) to O(N) for sequence length N.

*Tiling and dataflow*: Break large matrices into tiles that fit in SRAM, and order operations so the same tile is reused as many times as possible before eviction (maximizing arithmetic intensity). The roofline model is the standard tool: plot FLOPs/byte (arithmetic intensity) against hardware's compute-to-bandwidth ratio; operations below the ridge point are memory-bound.

*KV-cache management*: During LLM inference, past token key-value pairs must be cached and read back every step. For long contexts, this cache dwarfs the model weights in memory. Solutions include paging (PagedAttention), quantizing the cache to INT8/INT4, and offloading to CPU DRAM.

**Software stack**: The survey emphasizes that hardware potential is only realized through compilers (e.g., XLA, TVM, Triton) that perform layout transformations, tiling, and scheduling automatically, and runtime systems that handle dynamic batching, memory pooling, and load balancing across devices.

## Key Results

This is a survey paper rather than an empirical contribution, so it synthesizes results across the literature rather than reporting new benchmarks. Key quantitative anchors cited include:

- **Mixed-precision training** (FP16 compute / FP32 accumulate) delivers roughly 2–8× throughput improvement over FP32-only training with no meaningful loss in convergence quality.
- **INT8 quantization for inference** can reduce energy consumption and memory bandwidth by ~2–4× versus FP16, often with <1% accuracy degradation on standard benchmarks when quantization-aware training is used.
- **Structured 2:4 sparsity** (NVIDIA Ampere+) achieves ~2× speedup in matrix operations with modest accuracy trade-offs.
- **FlashAttention** reduces attention memory complexity from O(N²) to O(N) in sequence length and achieves 2–4× wall-clock speedup over standard attention for long sequences.
- **KV-cache** becomes the dominant memory consumer for sequences beyond a few thousand tokens, making memory-system design the central engineering challenge for LLM serving.
- **Data movement energy** dominates arithmetic energy in modern workloads — a recurring finding across multiple cited works — motivating the entire field of in-memory and near-memory computing.

## Limitations

Being a survey, the paper inherits the limitations of its scope decisions rather than producing falsifiable experimental claims. Several gaps are worth noting:

- **Benchmarking reproducibility** is explicitly flagged as an open problem: different accelerators are evaluated on different models, batch sizes, and precision settings, making apples-to-apples comparison nearly impossible.
- **Dynamic and sparse workloads** (MoE routing, variable-length sequences, sparse attention) are acknowledged as poorly served by current hardware, but the survey does not resolve this — it identifies it as a key open challenge.
- **Analog and neuromorphic** approaches receive relatively light treatment; the field is moving fast and many results are not yet reproducible at scale.
- **Security-aware deployment** (e.g., side-channel attacks on accelerators, model extraction) is mentioned as an open challenge but not deeply analyzed.
- **Energy measurement methodology** varies widely across papers, making it difficult to compare efficiency claims across accelerator families.

## Why It Matters

This survey arrives at an inflection point: LLMs have made hardware acceleration a mainstream concern rather than a specialist niche. The insight that *memory movement, not arithmetic, is the binding constraint* has practical consequences for every practitioner — it explains why techniques like FlashAttention, KV-cache quantization, and operator fusion have such outsized impact, and why simply buying a faster GPU often doesn't help as much as expected.

For the LLM ecosystem specifically, the survey crystallizes the design space around serving accelerators (LPUs), paged memory management (PagedAttention), and long-context efficiency — areas that are actively shaping how next-generation inference infrastructure is built. The unified taxonomy is also a practical tool for researchers deciding which hardware platform to target for a given workload, and for hardware designers understanding which software-level optimizations they need to support.

## See Also

[[Transformer]] · [[Attention Is All You Need]] · [[FlashAttention]] · [[Mixture of Experts]] · [[Quantization]] · [[Roofline Model]] · [[PagedAttention]] · [[Systolic Array]] · [[High-Bandwidth Memory]]
