---
created: "2026-06-03"
title: "Scaling Laws"
tags: [glossary, training, foundations]
tldr: "Loss decreases predictably as a power law with compute, data, and parameters — Chinchilla shows optimal training uses ~20 tokens per parameter."
---

## TL;DR
Model performance follows smooth power-law curves with scale, letting you predict loss before spending the compute.

## Intuition
Kaplan et al. (2020) showed that cross-entropy loss scales as `L ∝ N^-0.076` with parameters and `L ∝ D^-0.095` with data — clean power laws spanning orders of magnitude. This means if you plot loss vs. compute on a log-log scale, you get a straight line, and you can extrapolate where you'll land before training a single large model. The practical upshot: scaling is *predictable*, which turns model development from art into engineering.

Hoffmann et al. (2022) (Chinchilla) corrected Kaplan's under-training conclusion — GPT-3's 175B params were trained on only 300B tokens, but optimal scaling says you want ~20 tokens per parameter. Chinchilla (70B params, 1.4T tokens) beat GPT-3 on most benchmarks at 4× less compute. The law isn't just about making models bigger; it's about the *ratio* of data to parameters given a fixed compute budget.

## Why It Matters
- **Budget allocation**: Given a fixed FLOP budget C, optimal model size N* ∝ C^0.5 and tokens D* ∝ C^0.5 — you can compute the right model size before spending a dollar.
- **Emergent abilities caveat**: Scaling laws predict smooth loss curves, but capability on specific benchmarks can appear discontinuously — this tension between average loss and task performance is still poorly understood.
- **Architecture-agnostic signal**: Power-law scaling holds across [[Transformer]] and [[State Space Model]] architectures, making it a universal benchmark for comparing architectural efficiency (same loss, less compute = better).

## Related Concepts
[[Transformer]], [[Mixture-of-Experts|MoE]], [[RLHF]], [[Mamba]], [[State Space Model]]
