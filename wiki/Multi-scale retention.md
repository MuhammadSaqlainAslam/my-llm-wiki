---
created: "2026-05-20"
title: "Multi-Scale Retention (MSR)"
tags: [glossary, retnet, attention, multi-head, efficiency]
tldr: "RetNet's multi-head retention layer: h parallel retention heads each with a different fixed decay γ_h, giving the model multiple time scales to reason over simultaneously. Directly analogous to multi-head attention but with decayed linear attention."
aliases: [multi-scale retention, MSR, multi-head retention]
---

## TL;DR

Multi-Scale Retention (MSR) is the multi-head version of [[RetNet]]'s [[Retention mechanism]]. Each of $h$ heads uses a different fixed decay rate $\gamma_h$ — e.g., $\gamma_h = 1 - 2^{-5-h}$ — ranging from near-1 (very long memory) to smaller values (shorter memory). This lets the model simultaneously attend to different time scales, much like how multi-head attention lets different heads learn different relationship types.

## Intuition

A single retention head with $\gamma = 0.99$ gives the model roughly 100-token effective memory. A head with $\gamma = 0.5$ forgets quickly. By having many heads with a geometric progression of $\gamma$ values, the model gets multi-resolution temporal coverage: some heads for local context, some for medium-range, some for long-range.

The multi-scale design also gives the model more representational flexibility — different heads can specialize in different tasks (local syntax, long-range coreference, etc.) without needing content-based gating.

After computing retention in each head independently, MSR applies a group norm to each head's output (for stability in the absence of softmax), then concatenates and linearly projects.

## Why It Matters

- **It's the direct analogue of multi-head attention for retention.** The multi-head structure is familiar and the per-head specialization works similarly.
- **Different γ values give different effective memory horizons.** Instead of one time scale for the whole model, MSR covers the full range.
- **Group norm is critical.** Without it, retention heads with different γ values produce outputs at wildly different scales.

## Where It Appears in This Wiki

- [[RetNet]] — MSR replaces multi-head attention in every RetNet layer

## Key Formula or Pseudocode

```
γ values: γ_h = 1 - 2^{-5-h}  for h = 1..H
  e.g., H=16: γ ∈ {1-1/64, 1-1/128, ..., 1-1/2097152}

Per head:  Y_h = Retention(Q_h, K_h, V_h, γ_h)
Per head:  Y_h = GroupNorm(Y_h)   # crucial for stability
Output:    MSR(X) = Concat(Y_1..Y_H) W_O
```

## Related Concepts

[[RetNet]] · [[Retention mechanism]] · [[Chunkwise recurrent]] · [[Exponential decay]] · [[Transformer]]
