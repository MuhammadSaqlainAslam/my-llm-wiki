---
created: "2026-05-12"
title: "Kernel Fusion"
tags: [hardware, optimization, gpu, sram]
tldr: "Combine multiple consecutive operations into a single GPU/TPU kernel so intermediate tensors stay in [[SRAM]] and never touch [[HBM]]. The pattern behind FlashAttention, Mamba's [[Hardware-Aware Scan]], and most modern high-performance ML kernels."
---

# Kernel Fusion

Naive deep-learning code chains independent kernels: each op reads its inputs from HBM, writes its output to HBM, then the next op reads it back. For element-wise or reduction-heavy ops, the HBM round-trip dominates the cost — you waste 90% of your time on data movement. **Kernel fusion** merges $N$ such ops into a single kernel: load operands once, perform all $N$ ops in registers/SRAM, write only the final output. The savings can be 10–40× wall-clock for memory-bound chains. FlashAttention fuses softmax-with-matmul; the [[Hardware-Aware Scan]] fuses the entire SSM recurrence with its discretization; modern compilers (Inductor, XLA) do similar fusions automatically when they can. **[[Operator fusion]]** is the same idea at the compiler level.

## Where it appears

- [[Hardware Acceleration for Neural Networks]]
- [[Mamba]]
- [[Hardware-Aware Scan]]

---

*Related: [[HBM]] · [[SRAM]] · [[Operator fusion]]*
