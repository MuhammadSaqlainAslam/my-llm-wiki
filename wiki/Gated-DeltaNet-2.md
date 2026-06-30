---
created: "2026-06-24"
title: "Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention"
authors: "Hatamizadeh, Choi, Kautz"
year: "2026"
arxiv: "2605.22791"
tags: [linear-attention, ssm, architecture, efficiency]
tldr: "Generalizes Gated DeltaNet and Kimi Delta Attention by replacing their single scalar gate with separate channel-wise erase and write gates, achieving the strongest results among Mamba-2/3 and DeltaNet-family variants on language modeling and long-context retrieval"
citation_count: 5
---

## TL;DR

Gated DeltaNet-2 decouples the single scalar gate used by Gated DeltaNet and Kimi Delta Attention (KDA) into two separate channel-wise gates: an erase gate controlling which key-side coordinates of the old state are removed, and a write gate controlling which value-side coordinates of the new content are committed. At 1.3B parameters on 100B FineWeb-Edu tokens, it outperforms [[Mamba]] (Mamba-2), [[Mamba3|Mamba-3]], Gated DeltaNet, and KDA, with its largest advantage on long-context RULER needle-in-a-haystack retrieval.

---

## The Problem

KDA already made the *decay* channel-wise, but its active edit (the delta rule update) still ties erasing old content and writing new content to one shared scalar gate — even though erasing operates on the key-side read direction and writing operates on the value-side commit direction: two genuinely different decisions that should be controlled independently.

---

## The Idea

Introduce the **Gated Delta Rule-2**: replace the tied scalar $\beta_t$ with an independent channel-wise erase gate $b_t$ (applied to the key-side read) and write gate $w_t$ (applied to the value-side write). The update recovers KDA exactly when both gates collapse to the same scalar, and recovers Gated DeltaNet when the decay also collapses to scalar — so it is a strict generalization, not a separate design.

A chunkwise WY-form algorithm preserves efficient parallel training, with a gate-aware backward pass required since (unlike the scalar case) the gates cannot be factored outside the relevant matrix products.

---

## Why It Matters

- A mathematically specified generalization that strictly subsumes two established models (Gated DeltaNet and KDA) as special cases — not a reapplication of existing methods
- Largest gains appear specifically on interference-heavy long-context retrieval (RULER multi-key needle-in-a-haystack), exactly where a fixed-size compressed state is under the most pressure to selectively forget
- Ablations show the erase gate's channel-wise structure contributes more than the write gate's — a concrete empirical finding about where the expressiveness gain comes from
- Directly benchmarks against [[Mamba3|Mamba-3]] (SISO and MIMO variants), making it a natural cross-reference for that note

---

## Limitations

- Evaluated at 1.3B parameters / 100B tokens — results at frontier model scale are not reported in this paper
- Modest constant throughput overhead vs KDA (38.0 → 36.1 Kt/s at 16K sequence length on H100) from the added gate computation

---

## Related Concepts

*Architecture: [[Mamba]] · [[Transformers Are SSMs]] · [[Mamba3|Mamba-3]] · [[DeltaNet]] · [[Gated Linear Attention]]*

*Context: [[State-Space Models]] · [[Flash Attention]]*
