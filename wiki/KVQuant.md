---
created: "2026-06-11"
title: "KVQuant"
aliases: ["KVQuant"]
year: 2024
tags: [kv-cache, quantization, inference, stub]
tldr: "Per-channel and per-token quantization of KV cache to 2–4 bits; uses non-uniform quantization grids calibrated to the heavy-tailed activation distribution."
---

## TL;DR
KV cache activations have heavy tails — a few outlier channels are much larger than the rest. KVQuant uses non-uniform quantization (calibrated per-channel) to handle these outliers without sacrificing bit-budget, achieving 2-bit KV with <0.1 ppl degradation on LLaMA-2-70B.

## See Also
[[KV Cache]] · [[NVFP4]] · [[Quantization]] · [[GQA]]
