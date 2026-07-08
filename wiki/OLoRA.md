---
created: "2026-07-08"
title: "OLoRA: Orthonormal Low-Rank Adaptation of Large Language Models"
authors: "Kerim Büyükakyüz"
year: "2024"
arxiv: "2406.01775"
tags: [fine-tuning, lora, parameter-efficient, initialization, efficiency]
tldr: "Enhances LoRA by replacing random initialization of the low-rank matrices A and B with orthonormal initialization via QR decomposition — accelerates convergence while keeping the same parameter count and memory footprint as standard LoRA"
citation_count: 0
---

## TL;DR

OLoRA is a drop-in enhancement to [[LoRA Low-Rank Adaptation of Large Language Models|LoRA]] that replaces the standard random Gaussian initialization of LoRA's low-rank matrices with orthonormal initialization via QR decomposition. The change requires no additional parameters and no change to the training procedure — it only changes what values A and B start from. The result is faster convergence and improved final performance compared to standard LoRA across a range of language modeling tasks.

## The Problem

Standard [[LoRA Low-Rank Adaptation of Large Language Models|LoRA]] initializes its low-rank matrices with random Gaussian values. This initialization is not particularly principled — it doesn't encode any information about the base model's parameter space, meaning early training steps are spent "finding direction" before meaningful learning begins. Poor initialization is a known cause of slow convergence in deep learning generally; LoRA inherits this without addressing it.

## The Idea

Replace random Gaussian initialization with orthonormal initialization derived from the base model's pretrained weights via QR decomposition. Specifically, compute the QR decomposition of relevant weight matrices in the pretrained model to extract orthonormal basis vectors, and use these to initialize LoRA's A and B matrices. This gives the low-rank adaptation matrices a structured starting point aligned with the base model's learned representation space, rather than a random arbitrary one.

## Why It Matters

- A single, clean, principled change to LoRA that requires no architectural modification, no additional parameters, and no changes to the optimizer or training loop
- Directly addresses a known but often overlooked weakness: how you initialize the adaptation matrices matters more than LoRA's original paper acknowledged
- Complements the cross-model transfer direction explored by [[LoRA-X]] and [[Cross-LoRA]] from a different angle: those papers ask "how do we reuse adapters across models," OLoRA asks "how do we train better adapters from scratch"

## Limitations

- Relatively narrow set of evaluations from a single-author paper — independent reproduction at larger scale and across more diverse tasks would strengthen the claim
- QR decomposition of the base model's weights adds a one-time upfront cost at initialization — negligible for typical use but worth noting for very large models
- The improvement over standard LoRA is primarily in convergence speed rather than final asymptotic performance — if compute is unconstrained and training runs to convergence, the gap narrows

## Related Concepts

[[LoRA Low-Rank Adaptation of Large Language Models|LoRA]] · [[LoRA-X]] · [[Cross-LoRA]]
