---
created: "2026-05-20"
title: "Warp (GPU)"
tags: [glossary, hardware, gpu, parallelism, scheduling]
tldr: "A group of 32 threads that execute the same instruction in lockstep on a GPU. Inter-warp communication requires shared memory (SRAM); suboptimal warp scheduling was FlashAttention v1's bottleneck, fixed in v2."
aliases: [GPU warp, warp scheduling, SIMT warp]
---

## TL;DR

A warp is the fundamental unit of parallel execution on an NVIDIA GPU: 32 threads that run in lock-step, executing the same instruction simultaneously (SIMT — Single Instruction, Multiple Threads). The GPU schedules work in units of warps. When different warps need to share data, they communicate through shared memory (on-chip SRAM), which is fast but requires synchronization.

## Intuition

Think of a warp as a 32-lane highway where all cars must drive at the same speed and take the same exits. If all 32 threads are doing the same work (multiplying matrix tiles), the highway is efficient. If some threads wait while others finish (divergence), the highway wastes lanes.

Each GPU streaming multiprocessor (SM) can hold many warps simultaneously — 64 or more on an A100. This pool is the warp scheduler's budget. **Occupancy** measures what fraction of this budget is being used productively. Low occupancy = threads sitting idle = wasted GPU.

For [[Flash Attention|FlashAttention v1]], the backward pass divided work between warps in a way that forced inter-warp communication through shared memory unnecessarily: warp A would write a result to SRAM, warp B would read it, synchronize, proceed. This synchronization cost ate into throughput. [[FlashAttention-2]] restructured the work so each warp owns its output accumulator — no inter-warp sync needed — pushing GPU utilization from 25–40% to 50–73% of peak.

## Why It Matters

- **Warp scheduling is the main lever left after IO-awareness.** Once you've minimized HBM round-trips (FlashAttention v1), the next win is reducing inter-warp synchronization (FlashAttention v2).
- **Understanding warps explains the 2× gap between FA v1 and v2.** Same IO pattern, same algorithm — different warp assignment → 2× throughput improvement.
- **Warp divergence is a common performance bug.** Any `if` statement that different threads take differently causes half the warp to stall.

## Where It Appears in This Wiki

- [[FlashAttention-2]] — fixing suboptimal warp-to-warp communication in the backward pass is the central contribution
- [[Hardware Acceleration for Neural Networks]] — warp scheduling is discussed as a software-level optimization lever

## Key Formula or Pseudocode

```
A100 SM: 108 SMs × 64 warps/SM × 32 threads/warp = 221,184 threads peak

Occupancy = (active warps) / (max warps per SM)
- FlashAttention v1 backward: ~35% occupancy (inter-warp sync bottleneck)
- FlashAttention-2 backward: ~65% occupancy (each warp owns its accumulator)
```

## Related Concepts

[[FlashAttention-2]] · [[Occupancy]] · [[Thread block]] · [[SRAM]] · [[Flash Attention]] · [[Tensor Cores]]
