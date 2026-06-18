---
created: "2026-05-20"
title: "Online Softmax"
tags: [glossary, attention, numerical-stability, algorithm]
tldr: "A numerically stable formulation of softmax that accumulates the result incrementally over blocks, updating a running maximum and normalizer — enabling tiled attention computation without ever materializing the full attention matrix."
aliases: [online softmax, running softmax, incremental softmax]
---

## TL;DR

Standard softmax requires seeing all input values before dividing: $\text{softmax}(x)_i = e^{x_i} / \sum_j e^{x_j}$. This requires a complete pass over the row. Online softmax maintains a running maximum $m$ and accumulator $\ell$ that can be updated block-by-block, producing the same result without materializing the full vector. [[Flash Attention]]'s entire correctness argument rests on this identity.

## Intuition

Softmax has a numerical stability problem: $e^{x_i}$ overflows in float32 when $x_i > 88$. The standard fix is to subtract the row maximum before exponentiating: $\text{softmax}(x)_i = e^{x_i - \max_j x_j} / \sum_j e^{x_j - \max_j x_j}$. But this requires two passes: one to find the max, one to compute the normalized exponentials.

Online softmax solves this with a single-pass streaming algorithm. As you process each new block of values, you update two running statistics:
- $m$ — the running maximum seen so far
- $\ell$ — the running sum of $e^{x_j - m}$

When you encounter a new maximum, you rescale both the accumulator $\ell$ and the partial output $O$ by the correction factor $e^{m_\text{old} - m_\text{new}}$. After processing all blocks, $O$ holds the correct weighted average, and you divide by $\ell$ once at the end.

This is the mathematical core that lets [[Flash Attention]] tile the attention computation: each tile processes its block of $K$ and $V$ rows, updates $m$ and $\ell$, and accumulates into $O$ — all in SRAM. No global write of the $T \times T$ attention matrix required.

## Why It Matters

- **It's what makes FlashAttention correct.** The tiling approach would produce wrong answers without online softmax's correction factor when a new global max is discovered.
- **It enables exact attention at sub-quadratic memory.** Without this, any tiled attention algorithm would need to either approximate or make two passes over HBM.
- **It's the standard for attention kernels.** Every production attention implementation (PyTorch SDPA, cuDNN, JAX) uses online softmax internally.

## Where It Appears in This Wiki

- [[Flash Attention]] — uses online softmax to tile attention into SRAM
- [[FlashAttention-2]] — same algorithm, better scheduled across warps
- [[Tiling]] — online softmax is what makes tiling work for softmax operations

## Key Formula or Pseudocode

```
Initialize: m = -∞, ℓ = 0, O = 0

For each block of K, V (indexed j):
  S_j = Q @ K_j.T          # partial attention scores
  m_new = max(m, rowmax(S_j))
  ℓ_new = e^(m - m_new) * ℓ + rowsum(e^(S_j - m_new))
  O = e^(m - m_new) * O + e^(S_j - m_new) @ V_j
  m = m_new, ℓ = ℓ_new

Output: O / ℓ              # final normalized result
```

## Related Concepts

[[Flash Attention]] · [[Tiling]] · [[IO-awareness]] · [[SRAM]] · [[Recomputation]]
