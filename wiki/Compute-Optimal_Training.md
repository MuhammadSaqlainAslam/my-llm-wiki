---
created: "2026-06-03"
title: "Compute-Optimal Training"
tags: [glossary, training, scaling]
tldr: "Train the model size and dataset size in a fixed ratio that minimizes loss for a given compute budget — Chinchilla says ~20 tokens per parameter."
---

## TL;DR
For a fixed FLOP budget, there's an optimal balance between model size N and training tokens D — blowing all compute on a huge model undertrained on data is wasteful.

## Intuition
Kaplan et al. (2020) suggested scaling model size aggressively while keeping tokens roughly fixed. Chinchilla (Hoffmann et al., 2022) corrected this: with C FLOPs, you want N ≈ C^0.5 parameters and D ≈ C^0.5 tokens, yielding the famous ~20 tokens/param rule. GPT-3 (175B params, 300B tokens) was severely undertrained by this standard — a compute-optimal GPT-3 would be ~67B params trained on ~1.4T tokens, achieving the same loss at the same compute.

The key insight is that both N and D contribute diminishing returns to loss, and the marginal FLOP is better spent on data once the model hits a certain size. Llama models operationalized this by training smaller models on far more tokens (Llama-2 7B on 2T tokens), making them cheaper to *inference* while staying competitive — a deliberate departure from strict compute-optimality toward inference-optimality.

## Why It Matters
- **Budget allocation**: Tells you exactly how to split your compute budget between model size and data before training starts — wrong choices waste millions in GPU-hours.
- **Inference cost**: Compute-optimal models are often overparameterized for deployment; training smaller models longer (over-training beyond Chinchilla) reduces per-token inference cost at the expense of training efficiency.
- **Scaling law calibration**: Underpins all serious scaling law extrapolations — predicting loss at 10× compute requires knowing you're on the optimal frontier, not an arbitrary point.

## Related Concepts
[[Transformer]], [[Mixture-of-Experts|MoE]], [[RLHF]], [[GQA]], [[Speculative Decoding]]
