---
created: "2026-05-12"
title: "HBM (High Bandwidth Memory)"
tags: [hardware, memory, gpu, bandwidth]
aliases: ["High-Bandwidth-Memory"]
tldr: "Stacked DRAM chips placed next to the accelerator die through silicon interposers. ~3–8 TB/s aggregate bandwidth, dominant memory tier of modern GPUs and TPUs. Not as fast as on-chip [[SRAM]], but the only practical option for tens of GB of capacity."
---

# HBM (High Bandwidth Memory)

HBM is DRAM that has been *stacked* — multiple memory dies vertically piled inside a single package, connected to the accelerator via a wide silicon interposer (1024–4096 bit bus). The stacking gives bandwidth that conventional GDDR can't match: HBM3 hits ~3 TB/s per stack, HBM3e ~5 TB/s, HBM4 (B200) ~8 TB/s. But it's still DRAM — capacity in the tens of GB, latency in hundreds of cycles, energy per byte 10–100× higher than on-chip [[SRAM]]. Most modern GPU/TPU optimization is about *not* reading HBM: keep hot data in SRAM via [[Kernel fusion]], minimize KV-cache movement during decode, etc. When people say an LLM is "memory-bandwidth-bound", HBM bandwidth is what they're measuring against.

## Where it appears

- [[Hardware Acceleration for Neural Networks]]

---

*Related: [[SRAM]] · [[Memory hierarchy]] · [[Kernel fusion]] · [[KV Cache]]*
