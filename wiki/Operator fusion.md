---
created: "2026-05-12"
title: "Operator Fusion"
tags: [hardware, compiler, optimization]
tldr: "The compiler-level version of [[Kernel fusion]]: a deep-learning compiler (XLA, Inductor, TVM) automatically merges adjacent operators into a single fused kernel. Saves you from writing the fused kernel by hand."
---

# Operator Fusion

The hand-written FlashAttention or Mamba scan is what kernel fusion looks like when an expert engineer does it. Operator fusion is what happens when the compiler does the same thing automatically — you write a chain of high-level ops (matmul → bias-add → ReLU → dropout) and the compiler emits a single CUDA / Triton / XLA kernel that performs them all in one pass without HBM round-trips. Modern frameworks (PyTorch 2 with TorchInductor, JAX with XLA, TensorFlow with XLA, ONNX Runtime) all do operator fusion as a standard optimization pass. It captures most of the easy wins; for the remaining hard kernels (attention, scan, MoE routing), human-written fused kernels still beat the compiler.

## Where it appears

- [[Hardware Acceleration for Neural Networks]]

---

*Related: [[Kernel fusion]]*
