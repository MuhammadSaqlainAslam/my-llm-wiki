---
created: "2026-07-08"
title: "LoRA-X: Bridging Foundation Models with Training-Free Cross-Model Adaptation"
authors: "Farzad Farhadzadeh, Debasmit Das, Shubhankar Borse, Fatih Porikli"
year: "2025"
arxiv: "2501.16559"
tags: [fine-tuning, lora, parameter-efficient, transfer-learning, diffusion-models]
tldr: "Qualcomm/ANU's ICLR 2025 paper — the first training-free cross-model LoRA transfer method, restricting the adapter to the source model's subspace and applying it only where source/target layers show acceptable subspace similarity; validated on Stable Diffusion v1.5/SDXL"
citation_count: 7
---

## TL;DR

LoRA-X is the earliest training-free method for transferring a [[LoRA Low-Rank Adaptation of Large Language Models|LoRA]] adapter from one base model to a different one, without retraining and without access to the original (or synthetic) training data. It works by constraining the adapter to operate within the source model's subspace, then applying it selectively — only to target-model layers whose weight subspace is similar enough to the corresponding source layer. Validated on text-to-image diffusion models (Stable Diffusion v1.5, SDXL), not LLMs.

## The Problem

Once a foundation model is deprecated and replaced by a newer one, every [[LoRA Low-Rank Adaptation of Large Language Models|LoRA]] adapter trained against the old model becomes useless — reusing it on the new model requires retraining, which needs either the original training data or enough synthetic data to approximate its distribution. Original training data is frequently inaccessible for privacy or licensing reasons, and generating a faithful synthetic substitute is often impractical. This makes every base-model upgrade cycle expensive for anyone maintaining a library of task-specific adapters.

## The Idea

Because the target model's internals are unknown beyond its raw weights, LoRA-X constrains the transferred adapter to operate within the *source* model's subspace — the only subspace it has reliable information about. It then checks, layer by layer, whether the target model's weight subspace is similar enough to the source model's for the constrained adapter to be meaningfully applied there. The adapter is only inserted into layers that pass this subspace-similarity check; layers that diverge too much between source and target are skipped.

## How It Works

- Compute the subspace of the source model's LoRA-adapted weight matrices via SVD
- For each candidate layer in the target model, measure subspace similarity against the corresponding source layer
- Apply the (source-subspace-constrained) adapter only to target layers clearing an acceptable similarity threshold
- No gradient updates, no training data, no synthetic data generation — the entire transfer is a closed-form, training-free operation

## Key Results

- Effective training-free transfer demonstrated on Stable Diffusion v1.5 and Stable Diffusion XL (text-to-image diffusion, not language models)
- Accepted to ICLR 2025

## Why It Matters

- First to demonstrate that LoRA adapters can be moved between different base models with zero training and zero data access — the founding result for the "data-free LoRA transfer" line of work
- Directly cited by [[Cross-LoRA]] as prior work when extending the same underlying idea from diffusion models to LLMs
- Also spawned ProLoRA (same authors), which removes the subspace constraint entirely by decomposing LoRA weights into subspace and null-space components — though like LoRA-X it remains focused on text-to-image diffusion rather than LLMs

## Limitations

- Requires the adapter to stay within the source model's subspace, which is a real constraint — it doesn't fully exploit the target model's own subspace where the two diverge
- Selective per-layer application means some layers simply don't get the adapter transferred at all if subspace similarity is too low, capping how much of the source task knowledge survives the transfer
- Validated only on text-to-image diffusion models (Stable Diffusion v1.5, SDXL) — no LLM validation in this paper; that extension is left to later work like [[Cross-LoRA]]

## Related Concepts

[[LoRA Low-Rank Adaptation of Large Language Models|LoRA]] · [[Cross-LoRA]] · [[QLoRA]] · [[Parameter-Efficient Fine-Tuning]]
