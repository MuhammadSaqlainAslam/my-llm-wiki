---
title: "Compressed Sparse Attention (CSA)"
tags: [attention, compression, long-context, efficiency, deepseek]
aliases: [CSA, Compressed Sparse Attention]
tldr: "Compress every m tokens into one KV entry via a learned weighted sum, then use a cheap Lightning Indexer to pick only the top-k relevant compressed blocks for each query. Result: attention over n/m entries instead of n, with sparsity cutting it further. Local precision is preserved by a small sliding-window supplement."
theme: efficiency
---

# Compressed Sparse Attention (CSA)

> Introduced in [[DeepSeek_V4]], 2026

## The Core Problem

Standard attention attends every query to every key — $O(n^2)$ work and an $O(n)$ [[KV Cache]]. At 1M tokens, the KV cache alone exceeds 40 GB. The naive fix, truncating context, throws away information. The right fix is **compression**: instead of storing one KV entry per token, store one entry per *group* of tokens.

CSA does this in two stages: first compress the sequence, then be selective about which compressed blocks to attend to.

---

## How It Works

### Stage 1 — Token-Level Compression

Every $m$ consecutive tokens are weighted-summed into a single KV entry. Concretely, for a window of $m$ tokens, CSA computes a raw KV tensor $C \in \mathbb{R}^{n \times c}$ and a per-token weight tensor $Z \in \mathbb{R}^{n \times c}$ via learned projections. The weights $Z$ are passed through a row-wise softmax (with learned positional biases) to produce mixing coefficients, then used to blend the $C$ values into one compressed entry $C^\text{comp}_i \in \mathbb{R}^c$.

CSA actually uses an *overlapping* two-stream design: two sets of KV tensors $(C^a, C^b)$ with their windows staggered by $m/2$, so each compressed entry draws from $2m$ source tokens. This gives each block awareness of its boundary context. The sequence shrinks from length $n$ to $n/m$.

### Stage 2 — Lightning Indexer (Sparse Selection)

Not all $n/m$ compressed blocks are relevant to every query. The Lightning Indexer computes a cheap *index score* $I_{t,s}$ between query token $t$ and compressed block $s$ using low-rank query projections and ReLU-gated dot products, then selects the **top-$k$** blocks. The query attends only to those $k$ blocks — attention becomes $O(k)$ per token instead of $O(n/m)$.

### Stage 3 — Shared-KV MQA

Core attention is Multi-Query Attention (MQA): each compressed KV entry serves as both key and value (shared across all query heads). The output is grouped and projected back to the hidden dimension via a two-step grouped projection to keep the output projection cost manageable.

### Sliding Window Supplement

A query cannot attend to tokens within its own current compressed block (causality requires attending only to *past* blocks). To avoid losing fine-grained local information, CSA adds a small sliding window of the most recent $n_\text{win}$ uncompressed tokens alongside the selected compressed blocks.

---

## Key Numbers

| Parameter | Role |
|---|---|
| $m$ | Compression rate (tokens per KV entry) |
| $k$ | Number of top compressed blocks selected per query |
| $n_\text{win}$ | Sliding window size for local context |
| Effective context | $k + n_\text{win}$ entries per query (vs. $n$ in full attention) |

At 1M tokens, CSA + [[Heavily Compressed Attention|HCA]] together reduce the KV cache to **10%** of [[DeepSeek_V4|DeepSeek-V3.2]]'s size.

---

## Why It Works

The compression is *learned* — the model trains the weighting function $Z$ jointly with everything else. Unlike fixed striding or pooling, the softmax-weighted sum learns to preserve the most informative signal within each block. The indexer then learns which blocks actually matter for each query type. Both are differentiable end-to-end.

---

## Where It Appears

- **[[DeepSeek_V4]]** — primary architecture; CSA layers are interleaved with [[Heavily Compressed Attention|HCA]] layers throughout the model

---

## Related Concepts

- [[Heavily Compressed Attention]] — the companion attention type; heavier compression, no sparse selection, dense attention over the compressed result
- [[KV Cache]] — what CSA is compressing; understanding its growth motivates the design
- [[GQA]] — another KV compression technique (fewer KV heads); complementary to CSA
- [[Transformer]] — the standard attention CSA is replacing
- [[DeepSeek_V4]] — the model that introduced CSA
