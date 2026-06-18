---
created: "2026-06-11"
title: "SpecFusion"
aliases: ["SpecFusion"]
year: 2025
tags: [speculative-decoding, multimodal, inference, stub]
tldr: "Speculative decoding for multimodal models: separate lightweight draft heads for text vs. image tokens, verified jointly by the full model."
---

## TL;DR
SpecFusion extends speculative decoding to multimodal LLMs (text + image) by using modality-specific draft heads that propose tokens in parallel, then verifying both modalities in a single target model forward pass.

## See Also
[[Speculative Decoding]] · [[Multi-Token Prediction]] · [[Medusa]]
