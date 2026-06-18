---
created: "2026-06-11"
title: "LLM-PQ"
aliases: ["LLM-PQ"]
year: 2025
tags: [quantization, hardware, mixed-precision, stub]
tldr: "Mixed-precision LLM deployment on hardware accelerators: assigns different bit-widths per layer based on sensitivity, minimizing quality loss at a given memory budget."
---

## TL;DR
LLM-PQ profiles each layer's quantization sensitivity and assigns bit-widths (e.g. 2-bit for insensitive layers, 8-bit for critical ones) to hit a target memory footprint with minimal perplexity degradation.

## See Also
[[NVFP4]] · [[Quantization]] · [[FPGA]] · [[Hardware Acceleration for Neural Networks]]
