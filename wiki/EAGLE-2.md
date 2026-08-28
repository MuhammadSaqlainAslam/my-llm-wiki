---
created: "2026-08-02"
title: "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"
authors: "Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang"
year: 2024
arxiv: "2406.16858"
tags: [speculative-decoding, inference, draft-model, feature-level, throughput, efficiency]
aliases: [EAGLE-2, EAGLE 2]
tldr: "EAGLE's draft tree is static — it assumes acceptance probability depends only on tree position. EAGLE-2 exploits the fact that the draft model's confidence scores are well-calibrated, and builds a context-aware dynamic draft tree from them instead. 3.05-4.26x speedup, 20-40% faster than EAGLE-1, still lossless."
theme: inference-optimization
citation_count: 371
---

# EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees

## TL;DR

[[EAGLE]] drafts at the feature level and verifies candidates arranged in a tree, but that tree's shape is **static** — fixed in advance, implicitly assuming a draft token's acceptance probability depends only on where it sits in the tree, not on the actual content being generated. EAGLE-2 removes that assumption: it builds a **context-aware, dynamic draft tree** at each step, using the draft model's own confidence scores (which turn out to be well-calibrated estimates of acceptance probability) to decide where to expand the tree.

## The Problem

A speculative-decoding draft tree has a fixed branching structure in EAGLE-1 — the same shape regardless of what's actually being generated. But the real acceptance rate of a draft token isn't just a function of tree position; it's context-dependent. Some contexts are highly predictable (acceptance rate near 1 almost everywhere) and would benefit from a wider, deeper tree; others are uncertain and a large static tree wastes verification compute on branches that will mostly get rejected.

## The Idea

EAGLE's draft model already produces a confidence score for each candidate token as a side effect of drafting. EAGLE-2's key empirical finding is that these confidence scores **approximate the true acceptance rate with small error** — the draft model is well-calibrated. That means the information needed to build a *good* tree shape is already available for free from the draft model itself.

EAGLE-2 uses this to grow the draft tree **dynamically**: expand branches where the draft model is confident (likely to be accepted, worth spending verification budget on), and prune branches where it's not. The result is a tree shaped to the actual context at each step, rather than one fixed shape used everywhere.

## Key Results

- **3.05x–4.26x** speedup across three series of LLMs and six tasks.
- **20–40% faster** than EAGLE-1 at equivalent settings.
- Remains **lossless** — output distribution is unchanged, same guarantee as EAGLE-1.

## Why It Matters

This is a case where an existing byproduct of the draft model (its confidence scores) turns out to already contain the signal needed to fix a design limitation (the static tree) — no new model or additional training needed, just a smarter use of information already being computed. It's a purely algorithmic improvement layered on top of EAGLE's architecture.

## Limitations

- The dynamic tree construction adds runtime bookkeeping compared to a fixed tree shape, though the paper shows this overhead is more than offset by the speedup.
- Benefits depend on the draft model's confidence calibration holding up — a poorly-calibrated draft model would build a worse dynamic tree than a well-tuned static one.

## Related Concepts

[[EAGLE]] · [[EAGLE-3]] · [[Speculative Decoding]] · [[Medusa]] · [[Lookahead Decoding]]
