---
created: "2026-08-02"
title: "Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding"
authors: "Zachary Ankner, Rishab Parthasarathy, Aniruddha Nrusimha, Christopher Rinard, Jonathan Ragan-Kelley, William Brandon"
year: 2024
arxiv: "2402.05109"
tags: [speculative-decoding, inference, draft-model, medusa, throughput, efficiency]
aliases: [HYDRA, Hydra heads]
tldr: "Medusa's draft heads speculate each position independently, ignoring earlier tokens in the same candidate continuation. Hydra heads condition on those earlier tokens instead — a drop-in, sequentially-dependent replacement. Increases average accepted length by up to 0.46 tokens; the tuned Hydra++ variant reaches 1.31x over Medusa and 2.70x over autoregressive decoding."
theme: inference-optimization
citation_count: 110
---

# Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding

## TL;DR

[[Medusa]] speeds up decoding with lightweight draft heads attached to the base model, each independently predicting a token at a fixed future position. That independence is a limitation: each head only ever sees the base model's last verified hidden state, never the *other draft tokens already speculated earlier in the same candidate continuation*. Hydra heads fix this with a simple, drop-in change — make the heads **sequentially dependent**, so later heads can condition on tokens already proposed by earlier heads in the same draft.

## The Problem

Medusa's draft heads are **sequentially independent**: head $k$ predicts the token at position $t+k$ using only the base model's hidden state at position $t$ — it has no information about what heads $1, \ldots, k-1$ already speculated for positions $t+1, \ldots, t+k-1$. This throws away useful information: a candidate continuation is a coherent sequence, and knowing the earlier speculated tokens should make later predictions more accurate.

## The Idea

Replace standard Medusa draft heads with **Hydra heads**: the same lightweight-head architecture, but each head now also takes the **earlier tokens in the candidate continuation** as additional input, not just the base model's last hidden state. This is a drop-in replacement — same overall Medusa decoding framework, same verification procedure — just heads that see more context about the draft they're extending.

The tuned variant, **Hydra++**, adds three refinements on top of the base Hydra heads:
- Deeper draft-head MLPs
- A teacher-distillation training objective
- An extra transformer decoder layer to better encode the already-verified sequence

## Key Results

- Base Hydra heads increase average candidate continuation acceptance length by **up to 0.46 tokens** versus standard Medusa heads.
- **Hydra++** achieves **1.31x** throughput over Medusa decoding and **2.70x** over plain autoregressive decoding.
- In batched inference, Hydra outperforms Medusa at **all evaluated batch sizes**, not just single-sequence decoding.
- Concurrently and independently developed alongside [[EAGLE]] — both arrived at sequential dependence (in different forms: EAGLE at the feature level, Hydra at the draft-head level) as a way to improve draft accuracy, which the authors note as complementary evidence for the idea.

## Why It Matters

Hydra and EAGLE reached a similar conclusion from different starting points — Medusa's heads and EAGLE's feature drafting both improve once they stop treating each future position as independent of the others already being speculated. That convergence, arrived at independently, is a strong signal that sequential dependence is a fundamental lever for draft accuracy in speculative decoding, not an idiosyncrasy of one particular architecture.

## Limitations

- Hydra heads still operate within Medusa's overall tree-based framework — it improves head accuracy, not the underlying decoding/verification structure.
- Hydra++'s extra training-time refinements (distillation, added decoder layer) require additional training cost beyond base Hydra or standard Medusa heads.

## Related Concepts

[[Medusa]] · [[EAGLE]] · [[EAGLE-2]] · [[EAGLE-3]] · [[Speculative Decoding]]
