---
created: "2026-05-20"
title: "IO-Awareness"
tags: [glossary, hardware, attention, efficiency, memory]
tldr: "Algorithm design principle: optimize for data movement between memory tiers (HBM↔SRAM), not just raw arithmetic — because on modern GPUs, bandwidth is the binding constraint, not compute."
aliases: [IO-aware, IO awareness, memory-IO]
---

## TL;DR

IO-awareness is the insight that on modern GPUs, **data movement is more expensive than arithmetic**. Peak TFLOPS keeps doubling every two years; HBM bandwidth grows at half that rate. An algorithm that does twice as many FLOPs but half as many HBM round-trips can be 2–4× faster in wall-clock time. IO-aware design counts bytes moved, not just multiply-accumulate operations.

## Intuition

Imagine you need to multiply a large matrix. The naive approach: read everything from disk (HBM), compute, write results back. An IO-aware approach: load one tile into fast cache (SRAM), do as much work as possible there, then move to the next tile — minimizing how often you touch the slow tier.

For attention, the culprit is the $T \times T$ attention matrix. For $T = 8192$ with 32 attention heads, this is 8 GB per layer per training step — read twice (once for softmax, once for the value matmul). [[Flash Attention]] eliminates these reads entirely by computing softmax incrementally tile-by-tile entirely in [[SRAM]]. The arithmetic is identical; the bytes moved drop by 4–10×.

The same principle powers [[Hardware-Aware Scan]] in [[Mamba]]: the SSM recurrence is fused into SRAM to avoid writing intermediate states to HBM.

## Why It Matters

- **Unlocks longer context.** Standard attention at 32K sequence length is impractical not because of FLOPs but because the $T \times T$ matrix exceeds GPU memory. IO-awareness collapses the memory footprint to $O(T)$.
- **Explains why approximate methods often lose to exact IO-aware ones.** Linear attention approximations have better FLOPs but more HBM traffic; FlashAttention is exact but wins on wall-clock time.
- **It's the lens for reading all hardware papers.** Every claim in [[Hardware Acceleration for Neural Networks]] ultimately reduces to: are you moving the right bytes?

## Where It Appears in This Wiki

- [[Flash Attention]] — the canonical IO-aware algorithm; reduces attention's HBM traffic by 4–10×
- [[FlashAttention-2]] — further improves GPU utilization by fixing warp-level IO patterns
- [[Hardware-Aware Scan]] — Mamba's SSM kernel: fuses the recurrence so intermediates stay in SRAM
- [[Hardware Acceleration for Neural Networks]] — the survey's organizing claim is that bandwidth, not FLOPS, limits modern LLMs

## Key Formula or Pseudocode

```
IO complexity (standard attention):  O(T² / B_HBM)  HBM round-trips
IO complexity (FlashAttention):      O(T · d / B_SRAM) SRAM round-trips

Where B_HBM ≈ 1–3 TB/s, B_SRAM ≈ 20 TB/s (on-chip)
```

## Related Concepts

[[Flash Attention]] · [[Tiling]] · [[SRAM]] · [[HBM]] · [[Kernel fusion]] · [[Hardware-Aware Scan]]
