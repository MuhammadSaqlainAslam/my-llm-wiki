---
created: "2026-07-16"
title: "OLMoE: Open Mixture-of-Experts Language Models"
authors: "Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert, Yuling Gu, Shane Arora, Akshita Bhagia, Dustin Schwenk, David Wadden, Alexander Wettig, Binyuan Hui, Tim Dettmers, Douwe Kiela, Ali Farhadi, Noah A. Smith, Pang Wei Koh, Amanpreet Singh, Hannaneh Hajishirzi"
year: "2024"
arxiv: "2409.02060"
tags: [model-family, moe, open-source, scaling, efficiency]
tldr: "Fully open MoE model family (weights, data, code, training details) with 1B active / 7B total parameters — competitive with much larger dense models, released at a scale where the full training recipe is reproducible by the research community. ICLR 2025."
citation_count: 0
---

## TL;DR

OLMoE is a fully open [[Mixture-of-Experts]] language model: weights, training data, training code, and experimental details are all publicly released. The 1B-active/7B-total parameter model achieves competitive performance with 7B-parameter dense models while using only 1B active parameters per token. Unlike most MoE releases (Mixtral, Kimi K2, GLM-5), OLMoE is designed specifically for the research community to reproduce, study, and build on. ICLR 2025.

## Architecture

- 64 experts, top-8 routing per token (1B active / 7B total parameters)
- Standard sparse [[Mixture-of-Experts]] design with top-k token routing
- Deliberately avoids proprietary routing innovations, using a simple, well-understood design to maximize reproducibility
- Trained on 5T tokens

## Why It Matters

- The "fully open" distinction is meaningful: most frontier MoE releases leave some component opaque (weights but not data, or weights with limited training detail). OLMoE releases everything — the constraint on reproducibility is compute, not access
- Competitive with similarly-sized dense models at roughly the same inference cost (1B active params) but with 7B parameter capacity
- The fully released training data and experimental logs enable ablations that are impossible with less open releases — directly useful for researchers studying MoE scaling behavior
- Provides an open MoE data point for the broader [[Scaling Laws]] literature

## Limitations

- 1B active / 7B total is a relatively small scale compared to production MoE models (Mixtral 8x7B, [[Kimi K2]]'s 1T total) — full openness is the primary contribution, not frontier performance
- Top-8-of-64 routing is a specific design choice that may not generalize optimally to other scales or domains without tuning

## Related Concepts

[[Mixture-of-Experts]] · [[Mixtral]] · [[Kimi K2]] · [[GLM-5]] · [[Scaling Laws]]
