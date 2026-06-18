---
created: "2026-05-20"
title: "Retention Mechanism"
tags: [glossary, retnet, linear-attention, recurrence, efficiency]
tldr: "RetNet's core sequence mixing layer: a decayed causal attention substitute that supports three equivalent computation forms — parallel (train), recurrent (O(1) inference), and chunkwise (linear-complexity long sequences). The retention score between positions n and m is γ^(n-m) · q_n k_m^T."
aliases: [retention, retention mechanism, RetNet retention]
---

## TL;DR

The retention mechanism is [[RetNet]]'s replacement for attention. The "retention" score between query position $n$ and key position $m$ is:

$$
\text{Ret}_{nm} = \gamma^{n-m} \cdot q_n k_m^\top
$$

where $\gamma \in (0,1)$ is a fixed per-head decay. This is linear attention (no softmax) with a positional exponential decay. The key insight: this single formula supports three different ways to compute the same result — parallel (like a Transformer, for training), recurrent (like an RNN, for fast inference), and chunkwise (a hybrid for long sequences).

## Intuition

Standard attention computes $\text{Attn}_{nm} = \text{softmax}(q_n k_m^\top / \sqrt{d})$: content-driven, nonlinear, $O(T^2)$ memory. Retention replaces the softmax with a fixed positional decay: position $n$ pays attention to position $m$ with weight proportional to $\gamma^{n-m} q_n k_m^\top$. No softmax normalization, just a product and a decay.

Because there's no softmax, the computation is linear in the sequence: you can accumulate the retention output as a running sum with a decay factor. This gives the recurrent form: state $S_n = \gamma S_{n-1} + k_n^\top v_n$, output $= q_n S_n$.

The multi-scale variant (MSR) runs $h$ retention heads with different $\gamma$ values — giving the model different time scales to work with, analogous to different attention heads learning different relationship types.

## Why It Matters

- **It's the mechanism that achieves the "impossible triangle"** — training parallelism + good performance + cheap inference — by having three equivalent computation modes.
- **The group norm is critical for stability.** Because retention lacks softmax's normalization, a group norm after the retention output is needed to prevent gradient issues.
- **15.6× throughput at inference** compared to Transformer at 8K sequence length, because the recurrent form has no KV cache — just the fixed $d \times d$ state per head.

## Where It Appears in This Wiki

- [[RetNet]] — retention is the central contribution; the three computation modes (parallel/recurrent/chunkwise) are derived from the same retention formula
- [[Multi-scale retention]] — the multi-head version with different γ per head

## Key Formula or Pseudocode

```
Retention score: Ret(n, m) = γ^(n-m) · (q_n · k_m)   [for n ≥ m; 0 otherwise]

Parallel form:  Y = (QK^T ⊙ D) V     where D_nm = γ^(n-m) if n≥m else 0
Recurrent form: S_n = γ S_{n-1} + k_n^T v_n           [O(d²) state]
                y_n = q_n S_n
Chunkwise:      parallel within chunks, recurrent across chunks
```

## Related Concepts

[[RetNet]] · [[Multi-scale retention]] · [[Chunkwise recurrent]] · [[Exponential decay]] · [[Linear attention]] · [[RWKV]]
