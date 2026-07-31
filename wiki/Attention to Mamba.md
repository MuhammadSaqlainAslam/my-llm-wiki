---
created: "2026-07-30"
title: "Attention to Mamba: A Recipe for Cross-Architecture Distillation"
authors: "Abhinav Moudgil, Ningyuan Huang, Eeshan Gunesh Dhekane, Pau Rodríguez, Luca Zappella, Federico Danieli"
year: 2026
tags: [ssm, mamba, distillation, linear-attention, cross-architecture, efficiency]
tldr: "Two-stage distillation — Transformer to linearized attention via a kernel trick, then linearized attention to Mamba — lets a pure-Mamba student recover Pythia-1B teacher performance (14.11 vs 13.86 perplexity) without falling back to a hybrid Attention+SSM architecture."
theme: efficiency
arxiv: "2604.14191"
citation_count: 1
---

# Attention to Mamba: A Recipe for Cross-Architecture Distillation

## TL;DR

Distilling a Transformer straight into [[Mamba]] fails — the architectures are too different for the student to inherit the teacher's behavior directly. This paper inserts a stepping stone: first distill the Transformer into a **linearized-attention** model (same computation graph, kernelized softmax), then distill *that* into Mamba. The intermediate step gives Mamba a principled initialization, and the distilled 1B model nearly matches its Transformer teacher (14.11 vs. 13.86 perplexity) without needing any Attention blocks left in the final architecture.

## The Problem

The community has spent years learning how to train Transformers well, and there's a glut of pretrained Transformer checkpoints. Mamba and other SSMs are cheaper to run (linear-time, constant-memory decoding — see [[Mamba]], [[State-Space-Models|State-Space Models]]) but starting a new Mamba model from scratch throws away all of that accumulated Transformer investment.

The obvious fix is distillation: train a Mamba student to mimic a Transformer teacher. But prior work found that naive Transformer→Mamba distillation just doesn't preserve teacher quality. The usual workaround is a **hybrid** architecture that keeps a few Attention layers mixed in with SSM blocks — which works, but means you never actually escape attention's O(n²) cost, just dilute it.

## The Idea

The failure isn't distillation itself — it's the size of the architectural jump. Attention and Mamba compute fundamentally different functions of the sequence, so asking a Mamba student to match a softmax-attention teacher directly is a big, poorly-conditioned optimization problem.

The fix is to break the jump into two smaller, better-conditioned ones:

1. **Transformer → Linearized Attention.** Distill the original softmax-attention teacher into a model that swaps softmax for a kernel feature map (see [[Linear attention]]): $\text{softmax}(qk^\top) \to \phi(q)\phi(k)^\top$. This is a much smaller architectural change — the model is still "attention-shaped," just without the softmax nonlinearity — so the student can match the teacher closely.
2. **Linearized Attention → Mamba.** Linear attention is already mathematically close to a (rank-1) SSM recurrence — this is exactly the connection formalized in [[Transformers Are SSMs|the SSD framework]]. That closeness means the linearized-attention checkpoint can be used to give the target Mamba architecture a **principled initialization**, rather than starting the second distillation stage from scratch. From there, distilling into an adapted Mamba model — with no Attention blocks left at all — is a much smaller step than the original Transformer → Mamba jump.

## Key Results

- Teacher: Pythia-1B. Student: fully Mamba-based (no Attention blocks), same parameter scale.
- Distilled student perplexity: **14.11**, vs. teacher's **13.86** — a small, largely closed gap.
- Downstream task performance is preserved alongside the perplexity match.
- Ablations at 1B scale over 10B distillation tokens vary: the sequence-mixer architecture, model size (scaling analysis), and how distillation tokens are split between the two stages (sensitivity analysis) — the two-stage recipe holds up across these settings.

## Why It Matters

This gives a way to convert existing pretrained Transformer checkpoints into genuinely linear-time Mamba models — not hybrids — without eating the usual distillation quality tax. If it holds at larger scale, it's a cheap path to SSM-speed inference from a Transformer you already trained, rather than needing to pretrain a new SSM from scratch.

## Limitations

- Demonstrated at 1B scale / 10B tokens — the paper includes its own scaling analysis, but it isn't yet validated at the scale of today's largest production Transformers.
- Still requires running the full two-stage pipeline (two distillation passes) rather than a single-shot conversion.

## Related Concepts

[[Mamba]] · [[Linear attention]] · [[Transformers Are SSMs]] · [[State-Space-Models|State-Space Models]] · [[On-Policy Distillation]]
