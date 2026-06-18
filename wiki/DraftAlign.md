---
created: "2026-06-11"
title: "DraftAlign"
aliases: ["DraftAlign"]
year: 2025
tags: [speculative-decoding, alignment, draft-model, stub]
tldr: "Fine-tunes the draft model in speculative decoding to better match the target model's output distribution, increasing token acceptance rate."
---

## TL;DR
The acceptance rate in speculative decoding is bottlenecked by distribution mismatch between draft and target. DraftAlign trains the draft model with a KL-divergence loss against the target, pushing acceptance rate toward 90%+ on benchmarks.

## See Also
[[Speculative Decoding]] · [[Multi-Token Prediction]] · [[KL Divergence]] · [[Medusa]]
