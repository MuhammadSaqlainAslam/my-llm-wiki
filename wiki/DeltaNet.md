---
created: "2026-06-03"
title: "DeltaNet"
tags: [glossary, linear-attention]
tldr: "A linear recurrent model using delta rule updates to write associative memories, achieving ~O(1) inference cost while closing the quality gap with softmax attention."
---

## TL;DR
DeltaNet replaces softmax attention with a hardware-friendly recurrent update rule (the delta rule) that selectively erases and rewrites a fixed-size hidden state matrix, enabling constant-memory inference without sacrificing in-context retrieval.

## Intuition
Standard [[Attention]] grows the [[KV Cache]] linearly with sequence length — brutal at 100k+ tokens. DeltaNet instead maintains a single d×d state matrix W that acts as an associative memory. At each step it computes an error signal (delta = target − current retrieval), then does a rank-1 update: W ← W + β·(v − Wk)·kᵀ. This is the Widrow-Hoff delta rule from 1960 dressed in modern SSM clothing — the key insight is that "forget selectively" beats "forget uniformly" (as in vanilla linear attention).

Concretely, DeltaNet matches or beats Mamba on language modeling perplexity at 1.3B parameters while using a fully parallelizable chunk-wise training algorithm (chunk size ~64–256 tokens) that reaches ~60% of FlashAttention throughput. The fixed state size means inference memory is O(d²) regardless of context length, vs. O(n·d) for a full KV cache at sequence length n.

## Why It Matters
- **Efficient long-context inference:** O(1) memory per layer at generation time — no KV cache explosion, making 1M-token contexts tractable on a single GPU.
- **Better associative recall than linear attention:** The erase-then-write delta rule prevents key collisions that degrade vanilla linear attention, recovering retrieval quality that previously required softmax.
- **Hardware-friendly training:** Chunk-parallel formulation maps cleanly to matrix multiplications on TPUs/GPUs, unlike purely sequential RNN-style recurrences (e.g. early Mamba implementations).

## Related Concepts
[[State Space Model]], [[Mamba]], [[Attention]], [[FlashAttention]], [[KV Cache]]
