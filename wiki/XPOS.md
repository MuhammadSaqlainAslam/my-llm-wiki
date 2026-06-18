---
created: "2026-05-20"
title: "XPOS (Extrapolatable Position Embedding)"
tags: [glossary, attention, position-encoding, retnet, length-extrapolation]
tldr: "A relative position embedding used in RetNet that multiplies the query and key by an exponentially decaying scalar per dimension, giving the model better length extrapolation than standard RoPE when combined with the retention mechanism."
aliases: [XPOS, xPos, extrapolatable position embedding]
---

## TL;DR

XPOS (Sun et al., 2022) is a positional encoding that multiplies the query and key vectors by position-dependent scalars $\xi_m$ that decay exponentially with distance: the attention score between positions $m$ and $n$ includes a factor $\xi_m / \xi_n = \zeta^{n-m}$ for $\zeta < 1$. This decaying attention aligns naturally with [[RetNet]]'s retention mechanism, and the resulting model extrapolates better to sequences longer than those seen during training.

## Intuition

Standard [[RoPE]] rotates query/key vectors by position-dependent angles, encoding relative positions through the dot product. XPOS adds an exponential decay on top: the effective attention between positions $m$ and $n$ decays with $|m - n|$ even before the softmax. For [[RetNet]], this decay is the retention mechanism's $\gamma^{n-m}$ factor — XPOS provides the positional encoding that works harmoniously with this structure.

The practical effect: when the model is evaluated on sequences longer than its training context, the XPOS decay naturally suppresses very distant tokens (distance > training length), preventing out-of-distribution positional encodings from destabilizing the attention.

## Why It Matters

- **It enables RetNet to generalize to longer sequences than training.** The decay structure is self-regularizing at long distances.
- **It's designed for the retention mechanism.** Unlike RoPE (which targets softmax attention), XPOS is designed for linear attention with decay — matching RetNet's architecture.

## Where It Appears in This Wiki

- [[RetNet]] — XPOS is the positional encoding used for multi-scale retention

## Related Concepts

[[RetNet]] · [[Retention mechanism]] · [[RoPE]] · [[Transformer]] · [[Exponential decay]]
