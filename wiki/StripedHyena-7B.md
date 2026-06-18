---
created: "2026-06-03"
title: "StripedHyena-7B"
tags: [glossary, ssm, hybrid, architecture]
tldr: "A 7B hybrid model alternating convolutional (Hyena) and attention layers — achieved GPT-NeoX competitive performance with faster inference, released by Together AI in 2023."
---

## TL;DR
StripedHyena-7B (Together AI, Dec 2023) alternates Hyena convolutional layers with multi-head attention in a striped pattern, achieving transformer-competitive perplexity while reducing memory and compute at long sequences.

## Intuition
Full self-attention is O(n²). Hyena operators are long convolutions that run in O(n log n) via FFT. StripedHyena interleaves the two: most layers are Hyena (cheap, captures long-range patterns efficiently), with a few attention layers (expensive, but critical for in-context reasoning and recall). The "striped" pattern matches empirical findings that you only need ~15-20% attention layers to preserve quality.

## Why It Matters
- Demonstrates that hybrid architectures (attention + sub-quadratic ops) can match GPT-NeoX at 7B scale
- Precursor to a broader class of hybrid models ([[Jamba]], [[Griffin]], [[Mamba]] hybrids) that became dominant in 2024
- Released open-weights, enabling community research on hybrid sequence models

## Related Concepts
[[Jamba]] · [[Griffin]] · [[Mamba]] · [[Transformer]] · [[RWKV]] · [[Linear attention]]
