---
title: "Heavily Compressed Attention (HCA)"
tags: [attention, compression, long-context, efficiency, deepseek]
aliases: [HCA, Heavily Compressed Attention]
tldr: "Compress every m' tokens (where m' >> m in CSA) into a single KV entry, then do ordinary dense attention over the tiny compressed sequence. No sparse selection needed — the sequence is already short enough. Coarser than CSA but cheaper. Interleaved with CSA layers in DeepSeek-V4."
theme: efficiency
---

# Heavily Compressed Attention (HCA)

> Introduced in [[DeepSeek_V4]], 2026

## Intuition

[[Compressed Sparse Attention|CSA]] compresses $m$ tokens into one entry and then prunes further with a sparse indexer. HCA skips the sparse step entirely: just compress *so hard* that the resulting sequence is short enough for plain dense attention. If you go from 1M tokens to 1K compressed entries, a full $O(n^2)$ attention over 1K is trivial.

The tradeoff is resolution. CSA preserves more signal per entry (smaller $m$, plus it selects the best blocks). HCA uses a much larger compression rate $m' \gg m$ — each entry represents a much longer span — so fine-grained detail within a block is lost. For layers where broad contextual awareness matters more than precise token-level retrieval, HCA is the right tool.

---

## How It Works

### Compression

HCA computes original KV entries $C \in \mathbb{R}^{n \times c}$ and weight tensor $Z \in \mathbb{R}^{n \times c}$ from the hidden states via learned projections. Each window of $m'$ consecutive tokens is blended into one compressed entry using Softmax-weighted combination (with learned positional biases $B \in \mathbb{R}^{m' \times c}$):

$$C^\text{comp}_i = \sum_{j=m'i}^{m'(i+1)-1} S_j \odot C_j, \quad S = \text{Softmax}_\text{row}(Z + B)$$

This reduces the sequence from length $n$ to $n/m'$. Unlike CSA, there is no overlapping between adjacent windows — each block is independent.

### Dense MQA over Compressed Sequence

After compression, HCA performs standard Multi-Query Attention (MQA) over the full set of $n/m'$ compressed entries — no indexer, no top-k selection. Because $m'$ is large (much larger than CSA's $m$), $n/m'$ is small enough that dense attention is cheap.

Queries are produced in a low-rank manner (down-project → up-project) shared with the indexer query path in CSA. Output projection uses the same grouped strategy as CSA to manage projection cost.

### Sliding Window Supplement

Like [[Compressed Sparse Attention|CSA]], HCA adds a small sliding window of the most recent $n_\text{win}$ uncompressed tokens to preserve local fine-grained information that heavy compression would otherwise destroy.

---

## CSA vs HCA

| Property | CSA | HCA |
|---|---|---|
| Compression rate | $m$ (moderate) | $m' \gg m$ (heavy) |
| Sparse selection | Yes (top-$k$ Lightning Indexer) | No (dense over all blocks) |
| Resolution | Higher | Lower |
| Compute | Indexer overhead + sparse MQA | Dense MQA over tiny sequence |
| Best for | Precise long-range recall | Cheap broad context sweep |

[[DeepSeek_V4]] interleaves both: CSA layers for precision, HCA layers for cheap global sweeps. Together they reduce the KV cache to **10%** of DeepSeek-V3.2 at 1M tokens.

---

## Where It Appears

- **[[DeepSeek_V4]]** — interleaved with [[Compressed Sparse Attention|CSA]] throughout the model; HCA handles the "cheap global context" layers

---

## Related Concepts

- [[Compressed Sparse Attention]] — the companion technique; finer compression + sparse selection
- [[KV Cache]] — the memory structure HCA is shrinking
- [[GQA]] — orthogonal KV compression (fewer heads, not fewer tokens)
- [[Transformer]] — the standard attention HCA replaces in HCA layers
- [[DeepSeek_V4]] — the model that introduced HCA
