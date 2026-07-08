---
created: "2026-07-08"
title: "Time-Varying LoRA: Towards Effective Cross-Domain Fine-Tuning of Diffusion Models"
authors: "Zhan Zhuang, Yulong Zhang, Xuehao Wang, Jiangang Lu, Ying Wei, Yu Zhang"
year: "2024"
arxiv: ""
technical_report: "https://openreview.net/forum?id=SgODU2mx9T"
source_type: "conference_paper"
tags: [fine-tuning, lora, parameter-efficient, diffusion, domain-adaptation]
tldr: "Terra — a time-varying low-rank adapter that builds a continuous parameter manifold parameterized by a scalar t in [0,1], enabling smooth domain interpolation and cross-domain fine-tuning of diffusion models. NeurIPS 2024 main track."
citation_count: 0
---

## TL;DR

Terra (Time-varying low-rank adapter) introduces a continuous parameter manifold for [[LoRA Low-Rank Adaptation of Large Language Models|LoRA]] adapters, parameterized by a scalar time variable t ∈ [0,1]. At t=0 the adapter degenerates to a standard source-domain LoRA; at t=1 it behaves as a target-domain LoRA; at intermediate values it generates interpolated domains. This enables domain-flow generation, unsupervised domain adaptation, and domain generalization — all within a single adapter. Accepted at NeurIPS 2024 (poster).

## The Problem

Standard [[LoRA Low-Rank Adaptation of Large Language Models|LoRA]] adapters are trained for a fixed source domain and a fixed target domain. When the target domain changes, or when you need to interpolate between domains (e.g. generating images that gradually shift from one style to another, or adapting to an unseen target domain), you must either retrain from scratch or use ad-hoc interpolation of fixed adapter weights. Neither approach produces a principled continuous path through domain space.

## The Idea

Parameterize the LoRA adapter weights themselves as a function of t, using a square matrix that varies with t, rather than fixed low-rank matrices. Training sets t=0 for source-domain samples and t=1 for target-domain samples, with the constraint that the manifold connecting them is smooth and expressively continuous. The paper proves a theorem on equivariance between Terra and multiple independently-trained LoRAs, showing Terra can implement two task-specific LoRAs via a single parameter manifold with fewer total parameters.

Two main applications:
- **Domain flow generation**: sample any t ∈ [0,1] to generate images from intermediate domains, useful for data augmentation bridging source and target
- **Domain generalization**: training on the generated intermediate domains improves generalization to unseen target domains relative to training only on source and target

## Domain Note

Terra operates on **diffusion models for image generation**, not on large language models. The LoRA mechanism is the same but the modality and use case differ from [[Cross-LoRA]] and [[Adapters Selector]], which target LLMs. Worth distinguishing when citing.

## Why It Matters

- Introduces a genuinely new degree of freedom for LoRA: continuous parameter interpolation along a learned manifold, rather than the binary source/target distinction standard LoRA enforces
- NeurIPS 2024 main track — peer-reviewed, not a preprint
- Code publicly available (github.com/zwebzone/terra)

## Limitations

- Validated on image generation (diffusion models); transfer to LLM fine-tuning or other domains is not demonstrated in this paper
- The continuous manifold assumption may break down for domain pairs that are structurally very different — the paper's evaluated domain pairs are visually related (style, subject variations)
- Additional training complexity compared to standard LoRA — the time-conditioned parameterization requires careful choice of t scheduling

## Related Concepts

[[LoRA Low-Rank Adaptation of Large Language Models|LoRA]] · [[LoRA-X]] · [[Cross-LoRA]] · [[Adapters Selector]]
