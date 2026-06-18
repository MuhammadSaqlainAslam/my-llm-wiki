---
created: "2026-05-20"
title: "Time-Mixing (RWKV)"
tags: [glossary, rwkv, rnn, attention, efficiency, recurrence]
tldr: "The RWKV sublayer responsible for cross-time interactions. Implements the WKV (Weighted Key Value) mechanism: an exponentially decaying weighted sum of past key-value pairs, computable as a Transformer in parallel training or as an O(1) RNN at inference."
aliases: [time mixing, WKV mechanism, RWKV time-mixing]
---

## TL;DR

In [[RWKV]], each transformer-like block contains two sublayers: time-mixing (cross-time interactions, like attention) and [[Channel-mixing]] (within-position transformation, like the FFN). Time-mixing implements the WKV operation — a linear attention variant where the "attention weight" between position $n$ and $k$ is $e^{-(n-1-k)w + k_k}$, with $w \geq 0$ a learned per-channel decay. This downweights older tokens exponentially without any dot-product token comparison.

## Intuition

Standard attention computes how much token $n$ should attend to token $k$ using $e^{q_n \cdot k_k / \sqrt{d}}$ — a content-based similarity. RWKV's time-mixing replaces this with a positional decay: token $n$ attends to token $k$ with weight $e^{-(n-1-k)w}$ (exponentially decaying) modulated by $e^{k_k}$ (a key-like importance score). No dot product, no $T^2$ cost.

The result, called WKV (Weighted Key Value), is:

$$
\text{WKV}_t = \frac{\sum_{i<t} e^{-(t-1-i)w + k_i} v_i + e^{u+k_t} v_t}{\sum_{i<t} e^{-(t-1-i)w + k_i} + e^{u+k_t}}
$$

The special term $e^{u+k_t}$ for the current token (with a separately learned bias $u$) prevents the current token from being "pre-forgotten" by its own position.

Crucially, this sum can be computed as a running statistic: numerator $a_t = e^{-w} a_{t-1} + e^{k_t} v_t$, denominator similarly. This is why RWKV works both as a Transformer (parallel prefix sum at training) and as an RNN ($O(1)$ per step at inference).

## Why It Matters

- **It's the mechanism that makes RWKV both parallelizable and O(1) at inference.** The linear attention form allows both modes without approximation.
- **The learned decay $w$ is per-channel but input-independent.** This is RWKV's main limitation vs. Mamba (where decay depends on content).
- **It's one of the cleanest examples of linear attention at scale.** RWKV ran this to 14B parameters — proof the mechanism scales.

## Where It Appears in This Wiki

- [[RWKV]] — time-mixing is one of RWKV's two core sublayers; implements the WKV mechanism
- [[Linear attention]] — RWKV's WKV is a specific linear attention formulation with exponential positional decay

## Key Formula or Pseudocode

```python
# Time-mixing forward pass
r = W_r @ (μ_r * x + (1 - μ_r) * x_prev)   # receptance
k = W_k @ (μ_k * x + (1 - μ_k) * x_prev)   # key
v = W_v @ (μ_v * x + (1 - μ_v) * x_prev)   # value

wkv = compute_wkv(r, k, v, w, u)             # the WKV recurrence
output = W_o @ (sigmoid(r) * wkv)
```

## Related Concepts

[[RWKV]] · [[Channel-mixing]] · [[Time shift]] · [[Linear attention]] · [[Receptance]] · [[Exponential decay]]
