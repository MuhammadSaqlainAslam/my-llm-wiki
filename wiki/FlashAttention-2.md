---
created: "2026-05-20"
title: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
authors: "Tri Dao"
year: 2023
arxiv: "2307.08691"
citation_count: 3185
tags: [attention, hardware, kernel-fusion, gpu, parallelism, efficiency, warp-specialization]
tldr: "FlashAttention v2: fixes v1's suboptimal warp scheduling to push attention from 25–40% to 50–73% of peak GPU FLOPS — as close to GEMM efficiency as exact softmax gets. Reaches 225 TFLOPS/s per A100 training GPT-style models."
aliases: [FlashAttention-2, FlashAttention v2, FA2]
theme: hardware
---

# FlashAttention-2

> Tri Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", ICLR 2024 (arXiv:2307.08691)

## TL;DR

[[Flash Attention|FlashAttention v1]] showed that IO-awareness — tiling attention into [[SRAM]], never writing the $T \times T$ matrix to [[HBM]] — gives exact attention at 2–4× the speed. But v1 still only used 25–40% of the GPU's theoretical peak FLOPS. The culprit was bad **work partitioning**: suboptimal distribution of work across thread blocks and warps led to low occupancy and excess shared-memory reads/writes.

FlashAttention-2 is not a new algorithm. It's the same tiled online softmax, but with three targeted rewrites of the CUDA scheduling:

1. Fewer non-matmul FLOPs
2. Parallelism across thread blocks *per head*
3. Better warp-level work distribution inside each block

Result: **50–73% peak FLOPS** on A100, and **225 TFLOPS/s** end-to-end GPT training — approaching the efficiency of raw GEMM operations.

---

## The Core Idea — What Was Wrong with v1's Scheduling

A GPU executes work in thread blocks. Inside each block, 32-thread warps share fast registers and communicate through shared memory (SRAM). Good GPU kernels:

1. Keep all warps busy (high **occupancy**)
2. Minimize warp-to-warp communication through shared memory (expensive even on-chip)
3. Saturate the tensor cores with matmul work

FlashAttention v1's loop order exposed a bottleneck: for the backward pass especially, warps frequently had to synchronize on shared memory reads, burning cycles doing nothing. The attention FLOPs also included a lot of non-matmul work (softmax rescaling, elementwise ops) that tensor cores can't accelerate.

---

## Three Changes in v2

### 1. Reduce non-matmul FLOPs

Modern GPUs have 10–20× higher matmul throughput (tensor cores) vs. general floating-point (CUDA cores). Every non-matmul op is proportionally expensive. v2 rearranges the online softmax update to minimize elementwise rescaling:

- Old: rescale partial $O$ at every key block
- New: delay rescaling, apply it only once at the end of each query block

Concrete: v1 did $O(T \cdot d)$ rescaling operations per layer; v2 reduces this by roughly a constant factor depending on block size.

### 2. Parallelize over query blocks (forward pass)

v1 parallelized over batch × heads (outer loops), then ran the inner loop over key/value blocks sequentially per warp. For small batch or large $d$, there weren't enough outer-loop iterations to keep all thread blocks busy.

v2 adds a parallel dimension over **query blocks** in the forward pass: different thread blocks can process different chunks of the query simultaneously. This dramatically increases occupancy at long sequence lengths.

### 3. Better warp-level work partitioning (backward pass)

The backward pass computes $dQ$, $dK$, $dV$ and is more complex than the forward. v1 split work across warps in a way that required all warps to write to shared memory and synchronize. v2 reassigns which warp owns which accumulator, eliminating most shared-memory communication within a block.

---

## Key Concepts

- **[[Thread block]]** — the unit of GPU scheduling; each block has its own SRAM and synchronizes independently
- **[[Warp]]** — 32 threads that execute in lockstep; communication between warps goes through shared memory
- **[[Occupancy]]** — fraction of available warps that are active; low occupancy wastes GPU parallelism
- **[[Tensor cores]]** — matrix-multiply units inside GPUs; ~10–20× faster than scalar CUDA cores
- **[[Multi-head attention]] parallelism** — v2 parallelizes over sequence length (query chunks) in addition to batch and heads
- **[[Flash Attention|FlashAttention v1]]** — the foundation: tiled IO-aware attention; v2 is a better-scheduled implementation of the same algorithm

---

## Key Results

| Metric | FlashAttention v1 | FlashAttention-2 |
|---|---|---|
| Peak FLOPS utilization (A100, fwd) | 25–40% | **50–73%** |
| End-to-end training (GPT, A100) | ~150 TFLOPS/s | **225 TFLOPS/s** |
| vs. xFormers memory-efficient attention | ~2× faster | **~2× faster** |
| vs. FlashAttention v1 kernel | baseline | **~2× speedup** |

On H100 with FP8 support (v3 extension), throughput climbs further — but those results are from the separate FlashAttention-3 work.

Also supports **multi-query attention (MQA)** and **grouped-query attention (GQA)** natively, which share $K, V$ across heads — important for LLaMA 2, Mistral, and most production models post-2023.

---

## Comparison to Prior Work

- vs. **[[Flash Attention|FlashAttention v1]]** — same algorithm, better kernel. v2 is a strict drop-in improvement.
- vs. **cuDNN attention** — NVIDIA's cuDNN incorporated FlashAttention-style ideas in v8.9+; v2 was competitive or faster on the A100 at the time of release.
- vs. **FlashAttention-3** — v3 (2024) extends to Hopper (H100) with TMA (Tensor Memory Accelerator) and warp specialization for producer/consumer overlap, reaching ~1000 TFLOPS/s on H100 FP16.

---

## Limitations

- Still $O(T^2)$ FLOPs — no approximation, no linear scaling. For million-token contexts, the compute cost still grows quadratically.
- Hopper-specific features (TMA, async warp groups) require a separate implementation — addressed by FlashAttention-3.
- The implementation complexity is high (custom CUDA kernels); porting to non-NVIDIA hardware requires rewriting from scratch.

---

## Why It Matters

- **It's what almost every LLM trains with today.** PyTorch's `F.scaled_dot_product_attention`, xFormers, and vLLM all default to FlashAttention-2 (or its descendants) on NVIDIA hardware.
- **It removed attention as a training bottleneck for sequences up to ~32K.** Before v2, even with v1, training GPT-4 style 32K-context models was constrained by attention throughput. v2 largely fixed this.
- **It shows that kernel engineering is a first-class research contribution.** The algorithmic idea (tiled softmax) was in v1; the 2× improvement in v2 came purely from scheduling the same math better.

---

## Related Notes

[[Flash Attention]] · [[Transformer]] · [[Hardware Acceleration for Neural Networks]] · [[KV Cache Optimization]] · [[GQA]] · [[Kernel fusion]] · [[SRAM]] · [[HBM]] · [[Tensor Cores]]
