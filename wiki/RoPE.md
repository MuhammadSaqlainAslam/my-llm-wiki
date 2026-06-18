---
created: "2026-05-20"
title: "RoPE (Rotary Position Embedding)"
authors: "Su et al."
tags: [glossary, attention, position-encoding, llama2, transformer, efficiency]
arxiv: "2104.09864"
citation_count: 5000
tldr: "Encodes position information by rotating query and key vectors in 2D planes by an angle proportional to position. Relative position appears naturally in the dot product — the model sees how far apart two tokens are without explicit position tokens. Enables length extrapolation and is the standard for modern LLMs."
aliases: [RoPE, Rotary Position Embedding, rotary embeddings, rotary position]
---

## TL;DR

RoPE (Su et al., 2021) encodes position by rotating the query and key vectors for each attention head. Token at position $m$ has its query/key rotated by angle $m\theta_d$ in each 2D sub-dimension. When you compute $q_m \cdot k_n$, the dot product depends on $m - n$ (the relative position) rather than absolute positions — position information flows into attention naturally without explicit position tokens.

## Intuition

The original Transformer added learned or sinusoidal position embeddings to the token embeddings before attention. This works but has two problems: (1) the model sees absolute positions, not relative, so it's harder to generalize to longer sequences; (2) learned embeddings can't extrapolate beyond the training length.

RoPE is more elegant. For each pair of dimensions $(2i, 2i+1)$ in the $d$-dimensional key/query vector, rotate the 2D vector by angle $m \theta_i$ where $\theta_i = 1/10000^{2i/d}$ (the same base as sinusoidal embeddings, but applied as a rotation). Then:

$$q_m \cdot k_n = f(q, m) \cdot f(k, n) = \text{Re}[q_m \bar{k}_n]$$

where $\bar{k}_n$ is the complex conjugate. This product depends only on $q$, $k$, and $m - n$ — the relative position — not on $m$ or $n$ individually.

**Length extrapolation**: since the model learns relative positions, it can generalize beyond its training length by interpolating or scaling the base $\theta$. Community extensions like "yarn" and "rope-ntk" scale the base for 4× or 8× length extrapolation.

## Why It Matters

- **It's the standard position encoding for all modern open LLMs.** LLaMA, Mistral, Gemma, Phi, Falcon — all use RoPE.
- **Relative positions generalize better than absolute.** The model doesn't need to memorize "what position 512 looks like" — it just learns what "100 tokens ago" looks like.
- **It enables context length extension tricks.** By scaling the rotation angle (YaRN, NTK-RoPE), practitioners can extend a 4K-context model to 32K without full retraining.

## Where It Appears in This Wiki

- [[LLaMA 2]] — uses RoPE for all position encoding in the 7B–70B models
- [[Transformer]] — the alternative to sinusoidal or learned absolute position embeddings

## Key Formula or Pseudocode

```
For position m, dimension pair (2i, 2i+1):
  θ_i = 1 / 10000^(2i/d)
  R(m, i): rotation matrix by angle m·θ_i

  q̃_m = R(m) q_m    (rotate query at position m)
  k̃_n = R(n) k_n    (rotate key at position n)

Dot product: q̃_m · k̃_n = q_m · R(m-n) k_n  [depends only on m-n]
```

## Related Concepts

[[LLaMA 2]] · [[Transformer]] · [[GQA]] · [[Flash Attention]] · [[XPOS]]
