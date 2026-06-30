---
created: "2026-05-28"
title: "Chinchilla Scaling Laws"
authors: "Hoffmann et al."
tags: [glossary, scaling, training, llms]
arxiv: "2203.15556"
citation_count: 3442
tldr: "For a fixed compute budget, model size and training tokens should scale equally — overturning the Kaplan et al. belief that bigger models are always better."
---

## TL;DR
Given a compute budget C, the optimal strategy is to scale model parameters N and training tokens D proportionally: roughly N ∝ D, not the previous assumption of aggressively scaling N while undertraining.

## Intuition
Kaplan et al. (2020) suggested you should spend most of your compute on a larger model, training it for relatively few tokens. DeepMind's 2022 Chinchilla paper blew this up. They trained 400+ models across a wide range of sizes and token counts and found the optimal ratio is ~20 tokens per parameter. GPT-3 (175B params, 300B tokens) was massively undertrained by this metric — a 70B model trained on 1.4T tokens (Chinchilla itself) matched or beat it on most benchmarks.

The practical upshot is enormous: if you have a fixed FLOP budget, you should train a *smaller* model for *longer* rather than a huge model briefly. This reframed the entire field. Llama, Mistral, and most post-2022 models follow this — Llama-2 7B trains on 2T tokens (~285 tokens/param), consciously over-training relative to compute-optimal because inference is cheap and you want a smaller, fast model.

## Why It Matters
- **Inference efficiency**: A smaller, well-trained model is cheaper to serve at scale — compute-optimal training ≠ deployment-optimal training.
- **Benchmark recalibration**: Models previously ranked by parameter count had to be re-evaluated; a 7B Chinchilla-trained model can outperform a 70B undertrained one.
- **Training budget decisions**: Every serious LLM training run now uses Chinchilla ratios as a baseline, then deliberately over-trains on tokens for deployment economics.

## Key Formula or Mechanism
Optimal allocation given compute budget C (in FLOPs):

N_opt ∝ C^0.5
D_opt ∝ C^0.5

Concretely:
  N_opt ≈ (C / 6)^0.5      # ~6 FLOPs per token per param (fwd+bwd)
  D_opt = C / (6 · N_opt)

Rule of thumb: D_opt ≈ 20 × N_opt

Loss follows a power law:
**L(N, D) = E + A/N^α + B/D^β**
where E is irreducible entropy, α ≈ 0.34, β ≈ 0.28.

## Where It Appears
- **Hoffmann et al. (2022)** — *"Training Compute-Optimal Large Language Models"* (the Chinchilla paper)
- **Touvron et al. (2023)** — Llama 1 & 2 explicitly cite Chinchilla rationale for over-training smaller models
- **Kaplan et al. (2020)** — *"Scaling Laws for Neural Language Models"* (the prior work Chinchilla corrects)
- **Mistral 7B (2023)** — embodies the "overtrain small models" philosophy

## Related Concepts
[[Scaling Laws]]
[[Compute-Optimal Training]]
[[Kaplan Scaling Laws]]
[[Compute-Optimal Training|Training Token Budget]]
[[Speculative Decoding|Inference Efficiency]]
