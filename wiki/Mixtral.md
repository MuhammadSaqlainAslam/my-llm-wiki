---
title: "Mixtral"
authors: "Jiang et al."
arxiv: "2401.04088"
citation_count: 1946
tags: [glossary, moe, architecture, open-source, mistral]
tldr: "Mistral AI's mixture-of-experts language model: 8 expert FFN layers per transformer block, top-2 routing per token. Mixtral 8x7B matches LLaMA-2-70B quality at 5× fewer active FLOPs, and is fully open-weight."
aliases: [Mixtral, Mixtral 8x7B, Mistral MoE, "Mixtral-of-Experts"]
---

## TL;DR

Mixtral 8x7B (Jiang et al., 2024) applies [[Mixture-of-Experts]] (MoE) to the Mistral-7B architecture: each Transformer layer has 8 FFN experts, and each token is routed to the top 2 experts (total active parameters: ~13B per token, total parameters: ~47B). The result matches [[LLaMA 2]]-70B on most benchmarks while using 5× fewer active FLOPs per token. Mixtral was one of the first fully open-weight MoE models at this scale.

## Intuition

[[Mixture-of-Experts]] increases total model capacity (more parameters) without increasing per-token compute (routing sends each token to only 2 of 8 experts). Mixtral applies this to the standard Transformer FFN: the single FFN in each layer is replaced by 8 specialized FFNs (experts), with a learned gating function routing each token to its top-2 matches.

The practical effect: a model that "knows more" (47B total parameters worth of knowledge) while running at the speed of a 13B dense model per token. At the time of release, Mixtral matched GPT-3.5 on most benchmarks — remarkable for an open model.

## Why It Matters

- **It validated sparse MoE at the open-source level.** The Mixtral paper provided a fully reproducible, open-weight implementation of large-scale MoE, enabling community research.
- **It's the reference point for MoE-vs-dense comparisons.** When [[LLaMA 2]] evaluations compare against MoE models, Mixtral 8x7B is a standard baseline.
- **EAGLE tested on Mixtral 8x7B Instruct** — used as a challenging MoE inference acceleration benchmark.

## Where It Appears in This Wiki

- [[LLaMA 2]] — compared against in evaluations; Mixtral emerged as a strong competitor
- [[Mixture-of-Experts]] — Mixtral is a concrete realization of MoE principles
- [[EAGLE]] — tested on Mixtral 8x7B Instruct as a diverse model baseline

## Related Concepts

[[Mixture-of-Experts]] · [[LLaMA 2]] · [[Transformer]] · [[EAGLE]] · [[DeepSeek_V4]]
