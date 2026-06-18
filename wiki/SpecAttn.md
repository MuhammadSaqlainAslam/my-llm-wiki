---
created: "2026-06-11"
title: "SpecAttn"
aliases: ["SpecAttn"]
year: 2026
tags: [speculative-decoding, sparse-attention, inference, stub]
tldr: "Speculative decoding with sparse attention verification: the target model only runs attention on draft tokens that pass a sparsity-based pre-filter."
---

## TL;DR
SpecAttn reduces verification cost in speculative decoding by using a sparse attention pattern during the verification pass — only full attention on tokens flagged as uncertain by a cheap scoring step.

## See Also
[[Speculative Decoding]] · [[GQA]] · [[FlashAttention]] · [[Multi-Token Prediction]]
