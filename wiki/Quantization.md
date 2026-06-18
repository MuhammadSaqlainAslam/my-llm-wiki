---
title: "Quantization"
authors: ""
year: ""
arxiv: ""
tags: [glossary, quantization, efficiency, inference]
tldr: "Representing model weights and/or activations using fewer bits than the original training precision, trading a small accuracy cost for reduced memory and faster inference"
citation_count: 0
---

# Quantization

## TL;DR

Quantization reduces the numerical precision used to store and compute a model's weights and activations — for example, converting 16-bit floating point weights to 8-bit (FP8) or 4-bit ([[NVFP4]]) representations — to shrink memory footprint and increase inference throughput, generally at a small, controlled cost to accuracy.

---

## Intuition

A model trained in BF16 stores every weight using 16 bits. Post-training quantization maps those values onto a smaller set of representable numbers (e.g., 256 distinct levels for 8-bit, 16 for 4-bit), shrinking the model's memory footprint roughly in proportion to the bit reduction.

The central challenge is doing this without meaningfully degrading output quality. Naive quantization of certain sensitive components — some [[State-Space-Models|state-space]] or attention parameters — can cause disproportionate accuracy loss. This is why techniques like mixed-precision quantization, quantization-aware training/distillation, and component-specific handling (e.g., quantizing Mamba states separately with stochastic rounding) have become standard practice.

---

## Why It Matters

- Lets frontier-scale models like [[Nemotron_3_Super|Nemotron 3 Super]] and [[Nemotron_3_Ultra|Nemotron 3 Ultra]] ship FP8 and [[NVFP4]] checkpoints alongside full-precision weights, letting users trade accuracy for cost/speed
- [[NVFP4]] enables genuinely 4-bit *training*, not just post-training quantization — a meaningfully different and harder problem that Nemotron 3 demonstrates at production scale
- Increasingly co-designed with architecture choices: hybrid Mamba-Attention models need different quantization treatment for their recurrent state (stochastic rounding, FP16) vs. their attention layers (FP8) vs. expert GEMMs ([[NVFP4]])
- Works synergistically with [[KV Cache Optimization]] — quantizing the KV cache entries to INT8 or FP8 can multiply the effective context length before eviction is needed

---

## Related Concepts

*Applied in: [[Nemotron_3_Super|Nemotron 3 Super]] · [[Nemotron_3_Ultra|Nemotron 3 Ultra]]*

*Key format: [[NVFP4]] · [[KV Cache Optimization]]*

*Related efficiency: [[Flash Attention|FlashAttention]] · [[State-Space-Models|State-Space Models]]*
