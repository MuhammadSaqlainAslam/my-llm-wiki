---
created: "2026-05-20"
title: "Pallas Kernel"
tags: [glossary, hardware, tpu, jax, google, kernel-programming]
tldr: "Google's JAX-based language for writing custom GPU/TPU kernels, analogous to Triton for NVIDIA GPUs. Used by Griffin to implement an IO-aware RG-LRU scan kernel that keeps intermediate states in fast on-chip memory — enabling the hardware efficiency of FlashAttention's approach on TPU."
aliases: [Pallas, Pallas kernel, JAX Pallas]
---

## TL;DR

Pallas (Bradbury et al., 2018; extended for TPU v3+) is Google's low-level kernel programming interface for JAX, analogous to NVIDIA's Triton. It allows researchers to write custom computation patterns that control exactly which data lives in on-chip SRAM vs. off-chip HBM — critical for implementing IO-aware algorithms on TPU hardware.

[[Griffin]] uses Pallas to implement the RG-LRU scan: the recurrent state computation is fused into a Pallas kernel so that intermediate activations never touch HBM, analogous to how [[Flash Attention]] achieves IO-awareness via CUDA. Without this kernel, the RG-LRU would be memory-bound and slow despite its simple arithmetic.

## Why It Matters

- **It's the TPU equivalent of Triton/CUDA custom kernels.** To implement hardware-aware algorithms on Google's TPU stack, Pallas is the required tool.
- **It enabled Griffin's hardware efficiency claim.** Griffin on TPU-v3 matches Transformer training throughput largely because of the Pallas RG-LRU kernel.
- **It's increasingly the interface for custom JAX ops.** XLA has historically required HLO-level changes for custom kernels; Pallas provides a higher-level Python interface.

## Where It Appears in This Wiki

- [[Griffin]] — the efficient RG-LRU scan on TPU is implemented as a Pallas kernel

## Related Concepts

[[Griffin]] · [[RG-LRU]] · [[Diagonal recurrence]] · [[IO-awareness]] · [[Hardware-Aware Scan]] · [[Kernel fusion]]
