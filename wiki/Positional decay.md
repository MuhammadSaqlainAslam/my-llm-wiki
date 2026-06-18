---
created: "2026-05-20"
title: "Positional Decay"
tags: [glossary, retnet, rwkv, linear-attention, recurrence, position]
tldr: "The mechanism in RetNet and RWKV where attention weight between two positions decays exponentially with their distance — γ^(n-m) or e^{-dw} — regardless of content. Older tokens receive exponentially less influence, enabling O(1) recurrent inference."
aliases: [positional decay, position-based decay, distance decay]
---

## TL;DR

Positional decay assigns attention weights based purely on distance, not content: token at position $m$ receives weight $\gamma^{n-m}$ at output position $n$. This is the structural choice that lets [[RetNet]] and [[RWKV]] be computed as $O(1)$ recurrences: you can maintain a running sum $S_n = \gamma S_{n-1} + k_n^\top v_n$ rather than materializing the full attention matrix.

## Intuition

Content-based attention ([[Transformer]]) says: attend based on relevance — any token, any distance. Positional decay says: forget exponentially with distance — local tokens matter more. Both are valid inductive biases; positional decay is the right one when you want O(1) inference at the cost of long-range recall.

The decay rate determines the effective "context window":
- $\gamma = 0.99$: ~100-token half-life (remembers ~100 tokens clearly)
- $\gamma = 0.9$: ~10-token half-life
- $\gamma = 0.999$: ~1000-token half-life

[[RetNet]] uses per-head $\gamma$ (multi-scale), giving the model different effective windows per head. [[RWKV]] uses per-channel learned $w$ (more granular).

## Where It Appears in This Wiki

- [[RetNet]] — $\gamma^{n-m}$ is the fixed per-head positional decay in the retention mechanism
- [[RWKV]] — $e^{-(n-k)w}$ is the learned per-channel decay in the WKV mechanism
- [[Exponential decay]] — positional decay is the specific form of exponential decay used for attention-weight computation

## Related Concepts

[[RetNet]] · [[RWKV]] · [[Retention mechanism]] · [[Exponential decay]] · [[Linear attention]]
