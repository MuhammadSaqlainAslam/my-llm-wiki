---
created: "2026-06-11"
title: "TreeSpec"
aliases: ["TreeSpec"]
year: 2025
tags: [speculative-decoding, tree-search, inference, stub]
tldr: "Tree-structured speculative decoding: drafts a tree of candidate continuations instead of a single sequence, then verifies multiple branches in parallel to improve acceptance rate."
---

## TL;DR
Standard speculative decoding drafts one token sequence; if a token is rejected, all subsequent tokens are discarded. TreeSpec drafts a tree — multiple branching candidate sequences — and verifies all branches simultaneously, keeping whichever path the target model accepts deepest.

## Intuition
Think of it as beam search for the draft, but with a single verification pass that accepts the best valid prefix across all beams. Acceptance rate climbs because the target model can pick from multiple options rather than accepting or rejecting a single chain.

## Why It Matters
- Higher effective acceptance rate than single-chain speculative decoding at similar draft compute
- Verification is still one target-model forward pass (branches batched together)
- Particularly useful when the draft model is uncertain — trees cover ambiguous continuations

## See Also
[[Speculative Decoding]] · [[Multi-Token Prediction]] · [[Medusa]] · [[EAGLE]]
