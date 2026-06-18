---
created: "2026-05-20"
title: "Chunkwise Recurrent"
tags: [glossary, retnet, rwkv, efficiency, long-sequences, parallelism]
tldr: "A hybrid computation mode that processes sequences in fixed-size chunks — within each chunk using parallel attention-like computation, between chunks using an O(1) recurrent state update. Achieves linear complexity in sequence length while being GPU-friendly."
aliases: [chunkwise recurrence, chunk-parallel recurrence, chunkwise computation]
---

## TL;DR

Chunkwise recurrence is a third computation mode for [[RetNet]] (and similar linear-recurrence models) between the fully parallel (training) and fully recurrent (inference) modes. Divide the sequence into chunks of size $B$. Within each chunk, compute retention in parallel (fast matmuls). Between chunks, pass the recurrent state $S$ forward. Total complexity: $O(B \cdot T)$ — linear in sequence length, with a GPU-friendly $B^2$ inner loop per chunk.

## Intuition

Pure parallel training: $O(T^2)$ memory but uses fast matmuls — fine for short sequences. Pure recurrent inference: $O(1)$ memory but sequential, no GPU parallelism — fine for generating one token at a time. Chunkwise is the middle ground: chunk size $B$ controls where you sit on this tradeoff.

Each chunk of $B$ tokens is processed as a mini-Transformer (parallel retention within the chunk). Between chunks, the recurrent state $S_\text{chunk}$ is passed forward — it summarizes everything seen before the current chunk. The within-chunk computation can be formulated as:

$$Y_\text{chunk} = (\text{local retention within chunk}) + Q_\text{chunk} \cdot S_\text{prev}$$

where $S_\text{prev}$ is the recurrent state from all previous chunks. The first term is $O(B^2)$ matmul; the second is $O(B \cdot d^2)$ matmul — both GPU-efficient.

Typical chunk size: $B = 128$ or $B = 256$. This gives 16–64× more parallelism than pure token-by-token recurrence while keeping memory $O(T \cdot d^2 / B)$ instead of $O(T^2)$.

## Why It Matters

- **It's how RetNet trains on long sequences without quadratic memory.** Chunkwise makes training on 32K+ sequences tractable.
- **It generalizes naturally to other linear recurrence models.** RWKV uses a similar "chunked WKV" approach. Any model with a linear recurrence can use chunkwise computation.
- **It's between two extremes on every metric.** Faster than pure recurrence (parallelism), cheaper than pure parallel (memory), moderate on both.

## Where It Appears in This Wiki

- [[RetNet]] — chunkwise is the third computation mode; used for long-sequence training
- [[RWKV]] — similar chunked parallel scan used during training
- [[Transformers Are SSMs]] — the SSD framework describes a similar chunk-based computation for Mamba-2

## Key Formula or Pseudocode

```
Sequence length T, chunk size B, num chunks C = T/B

For chunk c = 1..C:
  Q_c, K_c, V_c = current chunk's projections

  # Intra-chunk: parallel retention (B × B matmul)
  Y_local = LocalRetention(Q_c, K_c, V_c, γ)    # O(B²)

  # Inter-chunk: apply accumulated state
  Y_cross = Q_c @ S_{c-1}                         # O(B × d²)

  # Update state for next chunk
  S_c = γ^B * S_{c-1} + K_c^T V_c                # O(d²)

  Y_c = Y_local + Y_cross

Total: O(B · T) — linear in T
```

## Related Concepts

[[RetNet]] · [[Retention mechanism]] · [[Multi-scale retention]] · [[RWKV]] · [[Transformers Are SSMs]]
