---
created: "2026-05-12"
title: "LPU (Language Processing Unit)"
tags: [hardware, llm, inference, low-latency]
tldr: "A new class of [[ASIC]] engineered specifically for LLM inference (Groq, SambaNova). Optimizes for low per-token latency by reducing dispatch overhead and providing predictable execution; often loses its advantage at large batch sizes where GPU GEMM utilization saturates."
---

# LPU (Language Processing Unit)

An LPU is an inference accelerator built around the workload pattern of a Transformer's autoregressive decode loop: one token at a time, full model weight sweep per token, latency-sensitive. Where a GPU dispatches kernels through CUDA + a CPU driver, an LPU keeps the entire control flow on-chip and removes the per-kernel launch overhead. Memory layout is also stripped down — Groq, for example, places all weights in on-chip SRAM with no HBM at all, trading capacity for predictable bandwidth. The result is dramatically lower per-token latency at small batch sizes (where GPU GEMMs underutilize their compute) and competitive throughput at moderate batches. At very large batches, GPU utilization catches up and the LPU advantage shrinks.

## Where it appears

- [[Hardware Acceleration for Neural Networks]]

---

*Related: [[ASIC]] · [[GPU]] · [[KV Cache]] · [[KV Cache Optimization]]*
