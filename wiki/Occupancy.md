---
created: "2026-05-20"
title: "GPU Occupancy"
tags: [glossary, hardware, gpu, performance, scheduling]
tldr: "The fraction of a GPU's available warps that are actively executing at any moment. Low occupancy means threads are idle waiting for memory or synchronization. FlashAttention-2's key fix was raising occupancy from ~35% to ~65% in the backward pass."
aliases: [GPU occupancy, warp occupancy, SM occupancy]
---

## TL;DR

Occupancy is the ratio of active warps to the theoretical maximum warps a GPU can sustain simultaneously. A100 can hold up to 64 warps per streaming multiprocessor (SM). If only 20 are active because the others are waiting for shared memory or HBM, occupancy is 31% — meaning 69% of the GPU is sitting idle.

## Intuition

High occupancy is how GPUs hide memory latency. When one warp stalls waiting for an HBM read (400–800 ns), the scheduler immediately switches to another warp that's ready to compute. This "latency hiding" only works if there are enough warps in flight. Low occupancy = not enough warps to hide the latency = visible stalls = poor throughput.

The causes of low occupancy:
1. **Register pressure** — kernels using many registers per thread leave fewer threads active per SM
2. **Shared memory pressure** — kernels using large shared memory blocks limit how many thread blocks fit per SM
3. **Synchronization barriers** — when all warps in a block must sync before proceeding, the SM stalls even if other warps exist

[[FlashAttention-2]] specifically addressed cause #3: the v1 backward pass had unnecessary synchronization between warps computing $dQ$, $dK$, $dV$. By restructuring which warp owns which output accumulator, v2 eliminated most of these barriers and raised backward pass occupancy from ~35% to ~65%.

## Why It Matters

- **It's the second-order optimization after IO-awareness.** Once you've minimized bytes moved (FlashAttention v1), the next win is keeping more warps active (v2).
- **It's often the gap between "fast kernel" and "production kernel."** Research kernels frequently have good algorithms but poor occupancy; production tuning fixes this.
- **NVIDIA's Nsight Compute** reports occupancy per kernel — it's the first metric practitioners check when a kernel underperforms.

## Where It Appears in This Wiki

- [[FlashAttention-2]] — raising backward pass occupancy from 35% to 65% is the core contribution
- [[Warp]] — warps are the scheduling unit whose active count determines occupancy
- [[Hardware Acceleration for Neural Networks]] — occupancy is discussed as a key GPU optimization metric

## Related Concepts

[[Warp]] · [[Thread block]] · [[FlashAttention-2]] · [[SRAM]] · [[Tensor Cores]] · [[GPU]]
