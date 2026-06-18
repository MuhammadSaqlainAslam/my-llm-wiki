---
created: "2026-06-11"
title: "TensorSSM"
aliases: ["TensorSSM"]
year: 2025
tags: [ssm, hardware, tensor-cores, acceleration, stub]
tldr: "Tensor-core-optimized SSM scan: reformulates the parallel prefix scan as batched GEMMs that land natively on Tensor Cores."
---

## TL;DR
Standard SSM scans are memory-bound and miss Tensor Core utilization. TensorSSM reformulates the recurrence as matrix operations that map to cuBLAS/cuDNN primitives, achieving near-peak throughput on A100/H100.

## See Also
[[Hardware-Aware Scan]] · [[State Space Models]] · [[Mamba]] · [[FlashAttention]]
