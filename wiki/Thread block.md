---
created: "2026-05-20"
title: "Thread Block (GPU)"
tags: [glossary, hardware, gpu, parallelism, scheduling]
tldr: "A group of warps (up to 1024 threads) that share on-chip SRAM and can synchronize with each other. Thread blocks are the unit of work dispatched to a GPU streaming multiprocessor. FlashAttention-2 parallelizes attention across thread blocks for higher throughput."
aliases: [CUDA thread block, GPU thread block, block]
---

## TL;DR

A thread block is a group of threads (organized into warps of 32) that share the same SRAM allocation and can synchronize via `__syncthreads()`. Each streaming multiprocessor (SM) runs one or more thread blocks simultaneously. Thread blocks are the coarsest unit of intra-SM parallelism. Across-block communication requires global memory (HBM) — which is why kernel design tries to make blocks independent.

## Intuition

The GPU execution hierarchy is: **thread → warp (32 threads) → thread block (multiple warps) → SM (multiple blocks) → GPU (multiple SMs)**.

Within a thread block, threads share SRAM and can synchronize cheaply. Across thread blocks, the only shared resource is HBM — slow. So kernel designers try to structure work so each thread block is self-contained.

[[FlashAttention-2]]'s key parallelism improvement was realizing that in the attention computation, different query blocks (rows of $Q$) can be processed by different thread blocks independently. FlashAttention v1 only parallelized over batch size and number of attention heads; v2 adds parallelization over sequence length chunks. More thread blocks active at once = higher SM occupancy = higher throughput, especially for small batch sizes or large head dimensions.

## Why It Matters

- **Block-level independence is the key to GPU scalability.** Algorithms that can be expressed as independent thread blocks scale linearly with GPU resources.
- **Thread block SRAM budget determines tile size.** A100 has 192 KB SRAM per SM; the tile sizes in FlashAttention are chosen to fit within this budget.
- **Cross-block dependencies require HBM synchronization** — the design constraint that makes FlashAttention's global-write-free approach valuable.

## Where It Appears in This Wiki

- [[FlashAttention-2]] — adding query-chunk-level parallelism across thread blocks is the main contribution
- [[Flash Attention]] — tiles sized to fit within a single thread block's SRAM
- [[Warp]] — thread blocks contain multiple warps
- [[SRAM]] — each thread block has private access to its SM's SRAM allocation

## Related Concepts

[[Warp]] · [[Occupancy]] · [[SRAM]] · [[Flash Attention]] · [[FlashAttention-2]] · [[GPU]]
