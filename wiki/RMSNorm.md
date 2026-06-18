---
created: "2026-06-03"
title: "RMSNorm"
tags: [glossary, normalization, training-stability]
tldr: "A simplified layer normalization that skips mean subtraction and just scales by RMS, saving ~10-15% compute over LayerNorm."
---

## TL;DR
RMSNorm normalizes activations by dividing by their root-mean-square instead of full mean/variance statistics, then applies a learned scale — same stability, less math.

## Intuition
Standard LayerNorm computes mean and variance, subtracts the mean (re-centering), then divides by std (re-scaling). RMSNorm asks: do you actually need re-centering? Empirically, no. Just divide each activation vector by its RMS (√(mean of squares)) and multiply by a learned γ. For a hidden dim of 4096, that's 4096 multiplications and a sqrt instead of two passes over the vector.

The hypothesis is that the re-scaling part of LayerNorm does the heavy lifting for training stability. The mean subtraction is largely wasted compute. Llama, Mistral, and most modern open LLMs switched to RMSNorm and saw no quality regression with measurable throughput gains. The learned γ per-dimension still gives the model full expressivity to rescale features as needed.

## Why It Matters
- **Speed**: Eliminates mean computation and subtraction — roughly 10-15% faster normalization, which matters when you apply it 2x per Transformer layer across hundreds of layers.
- **Ubiquity in frontier models**: Llama 1/2/3, Mistral, Gemma, and Qwen all use RMSNorm, so understanding it is table stakes for reading modern architecture papers.
- **Stability without overhead**: Achieves the same gradient flow benefits as LayerNorm (prevents activation explosion across deep residual stacks) with strictly fewer FLOPs — a clean Pareto improvement.

## Related Concepts
[[Transformer]], [[Attention]], [[RoPE]], [[GQA]], [[Mixture-of-Experts|MoE]]
