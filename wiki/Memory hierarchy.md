---
created: "2026-05-12"
title: "Memory Hierarchy"
tags: [hardware, memory, performance, gpu]
tldr: "The progression registers → SRAM → HBM → host RAM → SSD, each tier ~10× slower and ~10× larger than the previous. End-to-end performance is usually decided by which tier your hot data lives in, not by raw arithmetic throughput."
---

# Memory Hierarchy

Every modern accelerator has a strict memory hierarchy. On a GPU: thread-local registers (latency 1, capacity tiny) → on-chip [[SRAM]] / shared memory (latency ~10 cycles, capacity ~100 KB per SM) → L2 cache (~50 cycles, ~50 MB) → [[HBM]] (~500 cycles, ~80 GB) → host RAM (~10 µs, ~1 TB) → NVMe (~100 µs, ~10 TB). Each tier is 10–100× larger and 10× slower than the previous. The rule of thumb: **performance is set by where your data lives**. If your kernel reads HBM for every operand, you're HBM-bound. If you can stage operands into SRAM and reuse them across many ops, you're compute-bound — which is the regime you want.

## Where it appears

- [[Hardware Acceleration for Neural Networks]]

---

*Related: [[HBM]] · [[SRAM]] · [[Kernel fusion]]*
