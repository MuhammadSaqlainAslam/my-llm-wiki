---
created: "2026-05-12"
title: "TPU (Tensor Processing Unit)"
tags: [hardware, tpu, google, asic, systolic-array]
aliases: ["Tensor-Processing-Unit"]
tldr: "Google's custom ASIC for neural network training and inference, built around large [[Systolic array|systolic arrays]] for matrix multiplication. High matmul density and predictable execution; less programmable than [[GPU]]s."
---

# TPU (Tensor Processing Unit)

A TPU is structured around one or more $128 \times 128$ (or larger) [[Systolic array]]s — 2D grids of MAC units that pipeline operands through the grid one tile per clock. The systolic structure gets near-100% MAC utilization on large matmuls but is rigid: anything that doesn't decompose into matmul + element-wise ops runs poorly. TPUs use HBM and a high-bandwidth on-chip scratchpad (instead of GPU-style L1/L2 caches), and they're connected with the proprietary ICI inter-chip interconnect to form pods of thousands of devices. The programming model (XLA) is more constrained than CUDA but enables aggressive whole-graph optimization.

## Where it appears

- [[Hardware Acceleration for Neural Networks]]

---

*Related: [[Systolic array]] · [[GPU]] · [[ASIC]]*
