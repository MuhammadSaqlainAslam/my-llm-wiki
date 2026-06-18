---
created: "2026-05-12"
title: "Hardware-Aware Algorithms"
tags: [hardware, gpu, kernel-fusion, sram]
tldr: "Algorithms designed around the physical [[Memory hierarchy]] and parallelism of the target accelerator — keep operands in [[SRAM]], minimize [[HBM]] traffic, fuse ops. Examples: FlashAttention, Mamba's [[Hardware-Aware Scan]], paged KV cache."
---

# Hardware-Aware Algorithms

A *hardware-aware* algorithm is one whose design explicitly accounts for the fact that the cost of an operation on modern accelerators is dominated by *where its operands live*, not by how many flops it does. Three families dominate: (i) **fusion** — combine consecutive ops into one kernel so intermediate tensors stay in [[SRAM]] (FlashAttention, [[Hardware-Aware Scan]]); (ii) **tiling** — slice big tensors into SRAM-sized blocks, process one block at a time, write outputs back to [[HBM]] only at the end; (iii) **layout** — reorganize how data is laid out across the [[Memory hierarchy]] so accesses are contiguous and prefetcher-friendly (PagedAttention, FlashDecoding). The pattern is explicit in the [[Mamba]] and [[Transformers Are SSMs]] papers — same recurrence math, but the algorithm that runs it is co-designed with the GPU.

## Where it appears

- [[Mamba]]
- [[Transformers Are SSMs]]
- [[Hardware Acceleration for Neural Networks]]

---

*Related: [[Hardware-Aware Scan]] · [[Kernel fusion]] · [[Flash Attention]] · [[HBM]] · [[SRAM]]*
