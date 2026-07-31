---
created: "2026-07-08"
title: "Cross-LoRA: A Data-Free LoRA Transfer Framework across Heterogeneous LLMs"
authors: "Feifan Xia, Mingyang Liao, Yuyang Fang, Defang Li, Yantong Xie, Weikang Li, Yang Li, Deguo Xia, Jizhou Huang"
year: "2025"
arxiv: "2508.05232"
tags: [fine-tuning, lora, parameter-efficient, transfer-learning, efficiency]
tldr: "Data-free, training-free framework for transferring LoRA adapters between heterogeneous LLMs — LoRA-Align uses rank-truncated SVD to handle dimension mismatches, LoRA-Shift projects aligned updates into the target model's parameter space, runs in ~20 minutes on a commodity GPU"
citation_count: 8
---

## TL;DR

Cross-LoRA solves a fundamental limitation of [[LoRA Low-Rank Adaptation of Large Language Models|LoRA]]: a LoRA adapter trained on one base model cannot be reused on a different base model without retraining, because the two models may have different hidden dimensions, layer counts, and learned subspaces. Cross-LoRA transfers a LoRA adapter across architectures without any training data or fine-tuning, using two sequential steps — subspace alignment followed by parameter-space projection.

## The Problem

[[LoRA Low-Rank Adaptation of Large Language Models|LoRA]] is tightly coupled to the base model it was trained on. If you fine-tune a LoRA adapter on one model and then want to switch to a different base model — because of cost, licensing, or performance reasons — you cannot reuse that adapter without starting the fine-tuning process from scratch. This is wasteful: the adapter encodes task-specific knowledge that should in principle be transferable regardless of the underlying architecture.

[[LoRA-X]] (ICLR 2025) first demonstrated training-free, data-free LoRA transfer across base models, but only for text-to-image diffusion models (Stable Diffusion v1.5, SDXL), using a subspace-constrained adapter applied selectively per layer. Cross-LoRA extends the same underlying idea — data-free subspace alignment — to LLMs, using a different alignment mechanism (rank-truncated SVD + Frobenius-optimal projection instead of per-layer subspace-similarity filtering).

## The Idea

Two sequential, data-free components:

**LoRA-Align** — aligns the subspace of the source base model to that of the target base model. Uses rank-truncated singular value decomposition (SVD) to find the principal directions of both models' weight matrices, then learns a Frobenius-optimal linear transformation mapping one subspace to the other. Handles dimension mismatches (e.g. different hidden sizes) gracefully.

**LoRA-Shift** — takes the aligned subspace and projects the source LoRA's weight updates (the low-rank matrices A and B) into the target model's parameter space. The result is a new set of LoRA weight matrices that are compatible with the target architecture and carry the task knowledge from the source fine-tuning.

Both steps are closed-form or near-closed-form computations — no gradient descent, no training data, no GPU cluster required. The full transfer pipeline runs in roughly 20 minutes on a single commodity GPU.

## Key Results

- Up to 5.26% relative performance gain over the unmodified target base model across transferred tasks
- Validated across heterogeneous model pairs (different architectures, hidden dimensions, layer counts)

## Why It Matters

- Directly extends [[LoRA Low-Rank Adaptation of Large Language Models|LoRA]] into a use case the original paper did not address — architecture-agnostic adapter reuse
- Practical value for any team maintaining fine-tuned adapters who wants to switch base models without retraining from scratch — a common scenario as new frontier models release every few months
- Data-free is the critical property: most teams cannot freely redistribute their fine-tuning datasets (proprietary, licensed, privacy-sensitive), so a transfer mechanism that needs no data is far more deployable than one that does

## Limitations

- 5.26% relative gain means Cross-LoRA does not fully recover the performance of re-training the adapter natively on the target model — it is a practical approximation, not a lossless transfer
- Validated primarily on instruction-following and reasoning tasks; transfer quality on narrow domain-specific adapters (medical, legal, code) may differ
- Alignment quality degrades when source and target models are architecturally very different (e.g. very different hidden dimensions or attention configurations)

## Related Concepts

[[LoRA Low-Rank Adaptation of Large Language Models|LoRA]] · [[LoRA-X]] · [[Direct Preference Optimization Your Language Model is Secretly a Reward Model|DPO]] · [[RLHF|InstructGPT / RLHF]] · [[Ministral 3]] · [[Qwen3 Technical Report|Qwen3]] · [[GLM-5]]
