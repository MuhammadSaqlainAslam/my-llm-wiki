---
created: "2026-05-12"
title: "SRAM (Static Random-Access Memory)"
tags: [hardware, memory, on-chip, cache]
tldr: "On-chip cache memory built from flip-flops. Tiny by capacity (KB–MB) but 10–100× faster and lower-energy per byte than [[HBM]]. The tier you want your hot data to live in — every modern accelerator kernel design is about keeping operands in SRAM as long as possible."
---

# SRAM (Static Random-Access Memory)

SRAM stores each bit in a small flip-flop circuit (typically 6 transistors). It's volatile and area-expensive — that's why on-die SRAM is measured in MB, not GB — but it's blazingly fast: latency in single-digit cycles, bandwidth measured in TB/s per chip, energy per byte 100× lower than [[HBM]]. The hierarchy of register file → L1 → L2 → SRAM scratchpad → HBM is a series of capacity-vs-speed trade-offs. The discovery underlying FlashAttention, [[Hardware-Aware Scan]], and most modern kernel-fusion work is the same: the bottleneck isn't compute, it's HBM bandwidth — so the design rule is **load operands into SRAM once, do as much work as you can in SRAM, write the answer back to HBM**.

## Where it appears

- [[Hardware Acceleration for Neural Networks]]
- [[Mamba]]

---

*Related: [[HBM]] · [[Kernel fusion]] · [[Hardware-Aware Scan]]*
