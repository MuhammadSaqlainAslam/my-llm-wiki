---
created: "2026-05-12"
title: "GPU (Graphics Processing Unit)"
tags: [hardware, gpu, parallelism, tensor-cores]
tldr: "Massively parallel processor originally for graphics, now the dominant accelerator for deep learning. Modern variants (NVIDIA H100/B200, AMD MI300) add [[Tensor Cores]] for low-precision matmul and [[HBM]] for bandwidth."
---

# GPU (Graphics Processing Unit)

A GPU has thousands of small cores that run the same instruction in lockstep across many data elements (SIMT — single instruction, multiple thread). For matrix multiplication this is ideal: each output element is an independent dot product. Modern GPUs add specialized **[[Tensor Cores]]** that execute a small dense matmul (e.g., $4 \times 4 \times 4$) per clock per core, dramatically increasing throughput at low precision. They sit next to **[[HBM]]** memory (3–8 TB/s) which is fast by CPU standards but still 10–100× slower than on-chip [[SRAM]] — making [[Memory hierarchy]] management central to GPU performance. The dominant accelerator family for both training and inference of large neural networks.

## Where it appears

- [[Hardware Acceleration for Neural Networks]]

---

*Related: [[Tensor Cores]] · [[HBM]] · [[SRAM]] · [[Kernel fusion]]*
