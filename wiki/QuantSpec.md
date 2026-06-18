---
created: "2026-06-11"
title: "QuantSpec"
aliases: ["QuantSpec"]
year: 2025
tags: [speculative-decoding, quantization, inference, stub]
tldr: "Combines KV cache quantization with speculative decoding: a quantized draft model proposes tokens, the full model verifies, halving memory while maintaining speed."
---

## TL;DR
QuantSpec runs a 4-bit quantized draft model for speculation and a higher-precision target model for verification. The draft's low memory footprint means both models fit simultaneously, enabling speculative decoding without extra GPU memory cost.

## See Also
[[Speculative Decoding]] · [[Quantization]] · [[KV Cache]] · [[Multi-Token Prediction]]
