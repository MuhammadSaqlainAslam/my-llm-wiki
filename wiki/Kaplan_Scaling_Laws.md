---
created: "2026-06-03"
title: "Kaplan Scaling Laws"
authors: "Kaplan et al."
tags: [glossary, scaling, training]
arxiv: "2001.08361"
citation_count: 8349
tldr: "Loss decreases as a smooth power law with model size (N), data (D), and compute (C), with exponents ~0.07, 0.095, and 0.05 respectively."
---

## TL;DR
Neural language model loss follows predictable power laws with compute, parameters, and data — letting you forecast performance before spending a dollar on training.

## Intuition
Kaplan et al. (OpenAI, 2020) showed that test loss scales as L(N) ∝ N^(-0.076) and L(C) ∝ C^(-0.050) across many orders of magnitude — no phase transitions, no surprises, just clean log-log lines. The punchline: given a fixed compute budget, you should scale model size faster than data (roughly N ∝ C^0.73), which means undertrain a large model rather than fully train a small one.

This directly shaped GPT-3 (175B params, arguably undertrained on data). The laws also revealed that architecture details — depth vs. width, heads, etc. — matter far less than raw parameter count, as long as you're not wildly out of distribution. Chinchilla (2022) later revised the data coefficient upward, suggesting 1:1 token-to-parameter ratio is more compute-optimal than Kaplan's ~20:1 model-heavy prescription.

## Why It Matters
- **Budget allocation**: Power law exponents let you extrapolate a 10× compute run from a 100× cheaper pilot — companies use this to decide model size before a single large training run.
- **Architecture-agnostic signal**: Loss curves obey the same laws across Transformers of wildly different shapes, making N and C the universal levers rather than hyperparameter tuning.
- **Chinchilla correction**: Kaplan underweighted data; Chinchilla's revised laws (~equal scaling of N and D) reshaped the entire industry toward data-efficient models like Llama and Mistral.

## Related Concepts
[[Transformer]], [[Mixture-of-Experts|MoE]], [[RLHF]], [[Attention]], [[Speculative Decoding]]
