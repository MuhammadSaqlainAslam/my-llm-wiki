---
created: "2026-09-01"
title: "Priming: Hybrid State Space Models From Pre-trained Transformers"
authors: "Aditya Chattopadhyay, Elvis Nunez, Prannay Kaul, Benjamin Bowman, Evan Becker, Luca Zancato, David Thomas, Wei Xia, Stefano Soatto"
year: 2026
arxiv: "2605.08301"
tags: [hybrid-architecture, state-space-models, distillation, knowledge-transfer, ssm, inference-efficiency]
citation_count: 0
tldr: "Turns hybrid Transformer/SSM architecture design from a pretraining problem into a knowledge-transfer problem: initialize a Hybrid model from a pretrained Transformer, then recover downstream quality with under 0.5% of the source model's pretraining token budget. Enables the first controlled, apples-to-apples comparison of SSM layer types at scale."
aliases: ["Priming", "Priming Hybrid State Space Models"]
---

# Priming: Hybrid State Space Models From Pre-trained Transformers

> Aditya Chattopadhyay, Elvis Nunez, Prannay Kaul, Benjamin Bowman, Evan Becker, Luca Zancato, David Thomas, Wei Xia, Stefano Soatto (AWS Agentic AI), "Priming: Hybrid State Space Models From Pre-trained Transformers", May 2026 (arXiv:2605.08301)

## TL;DR

Hybrid Transformer/SSM models — some layers full attention, some layers a linear-time recurrence — give you eidetic memory where you need it and compressed fading memory everywhere else: smaller KV caches, faster decoding, and a much richer design space than a pure Transformer or a pure SSM. The catch: exploring that design space (which SSM variant? what mixing ratio?) has required training every candidate from scratch, so almost nobody outside a few labs with huge compute budgets can actually run the experiment. Priming reframes hybrid design as a *knowledge-transfer* problem instead of a pretraining problem: take an already-pretrained Transformer, swap some of its attention layers for SSM layers, and recover the original quality with a short alignment + post-training phase that costs **less than 0.5% of the source model's original pretraining token budget**. This makes it cheap enough to run controlled comparisons across SSM types, and the resulting Hybrid 32B model beats its Transformer source on long-context reasoning while decoding up to 2.3× faster.

## The Problem / Motivation

Hybrid architectures — interleaving standard softmax attention layers with recurrent SSM layers ([[Mamba]], [[Gated_DeltaNet_(Yang_et_al._2025)|Gated DeltaNet]], and their relatives) — are attractive because attention and SSMs fail in complementary ways. Attention keeps an exact, growing KV cache: perfect recall, but linear memory growth and quadratic compute. SSMs compress the whole history into a fixed-size state: constant memory and fast decoding, but they can forget details that a later query needs. Interleaving the two gets most of the throughput win with much less of the recall loss (this is the same bet made by [[Jamba]], [[Zamba]], [[Samba]], and other production hybrids already in this wiki).

The problem is empirical, not conceptual: *how* should you interleave them, and *which* SSM variant should you use? Answering that requires training multiple full-scale models to convergence under matched conditions — an experiment that costs millions of dollars at 30B+ parameter scale. As a result, large-model hybrid research has stayed confined to whatever narrow set of configurations a handful of well-funded labs happened to try, rather than a systematic sweep.

## The Idea

Don't train the hybrid from scratch — **prime** it from a Transformer that's already been trained. Take a pretrained source Transformer (Qwen, Llama, Mistral — Priming is architecture-family-agnostic, and works for both dense and MoE sources), replace a subset of its attention layers with SSM layers initialized to approximate the attention layers they replace, then run a short two-phase recovery:

1. **Alignment phase** — a brief distillation-style phase that teaches the newly-inserted SSM layers to reproduce the behavior of the attention layers they replaced, using the frozen source model's activations as a target.
2. **Post-training phase** — a short continued-training run (instruction tuning / long-context extension) on the now-hybrid model to close the remaining quality gap.

Both phases together cost under 0.5% of what it took to pretrain the source Transformer in the first place. Because the recipe is cheap, the authors can afford to hold everything else fixed and vary only the SSM layer type — turning "which SSM should I use in a hybrid?" from a rhetorical question into a controlled experiment.

## Architecture / Method

```
Pretrained Transformer (source)
   [Attn] [Attn] [Attn] [Attn] [Attn] [Attn] ...
        │
        │  select a subset of Attn layers to replace
        ▼
   [Attn] [SSM ] [Attn] [SSM ] [Attn] [SSM ] ...     ← Hybrid, freshly initialized SSM layers
        │
        │  Phase 1: Alignment
        │  SSM layers learn to mimic the replaced Attn layers'
        │  behavior, using the frozen source Transformer as teacher
        ▼
   [Attn] [SSM*] [Attn] [SSM*] [Attn] [SSM*] ...     ← aligned Hybrid
        │
        │  Phase 2: Post-training
        │  short instruction-tuning / long-context extension run
        ▼
   Primed Hybrid model                                 ← < 0.5% of source's pretrain tokens, total
```

Because Priming only touches the SSM-layer initialization and runs a short recovery, everything about the source Transformer's tokenizer, embeddings, and remaining attention layers is reused as-is. This is what makes the controlled SSM-type comparison possible: with the recipe cost fixed and cheap, the authors trained matched Hybrids using three different SSM layer types — **Gated KalmaNet (GKA)**, **Gated DeltaNet (GDN)**, and **Mamba-2** — under otherwise identical conditions.

## Key Results

| Comparison | Result |
|---|---|
| SSM expressiveness hierarchy (long-context reasoning) | GKA > GDN > Mamba-2 — and this ranking directly predicts downstream task performance |
| Hybrid GKA 32B vs. source Qwen3-32B | **+3.8** average reasoning points |
| Hybrid GKA 32B vs. a Transformer post-trained on the same data | within **1%** |
| Hybrid GKA 32B decode throughput vs. source Transformer | up to **2.3×** higher |
| Priming cost vs. source model's original pretraining budget | **< 0.5%** of tokens |
| Scale tested | 8B and 32B, native 128K context |

The headline finding isn't just "hybrids can match Transformers cheaply" — it's that Priming makes SSM-type comparison *tractable*, and the resulting expressiveness ranking (GKA > GDN > Mamba-2) gives architecture designers a data-driven default instead of a folk-wisdom guess.

## Comparison to Prior Work

- vs. **training hybrids from scratch (Jamba, Zamba, Samba)** — those papers each committed to one SSM choice and one mixing ratio for an entire pretraining run; Priming makes the SSM choice itself an experimental variable by removing the from-scratch cost.
- vs. **[[Attention to Mamba]]** (cross-architecture distillation already in this wiki) — both use a pretrained Transformer as a starting point, but Attention to Mamba targets a *pure* Mamba student via a two-stage linearized-attention bridge; Priming targets a *hybrid* (attention layers kept, not fully removed) and is explicitly designed to be SSM-type-agnostic so it can compare several recurrent mechanisms head-to-head.
- vs. **[[Transformers Are SSMs]]** (the SSD duality) — that work shows attention and selective SSMs share a common mathematical structure; Priming is a practical consequence of taking that closeness seriously — if the two families are structurally close, converting one into the other should be cheap, and this paper shows it is.

## Limitations

- Priming still requires access to the *frozen source Transformer* as a teacher during alignment — it's not a from-scratch recipe, it's a conversion recipe.
- The SSM-expressiveness ranking (GKA > GDN > Mamba-2) was established under Priming's specific recovery recipe; it's not guaranteed to hold for models trained from scratch with a different SSM under different data/optimizer conditions.
- Layer-replacement ratio (how many attention layers to swap, and which ones) is itself a design choice the paper doesn't claim to have fully optimized — it demonstrates the method works well at the ratios tested, not that those ratios are optimal.

## Why It Matters

Priming turns an expensive architecture-search question — "which SSM, and how much of it?" — into a cheap, repeatable experiment that any team with an existing pretrained Transformer can run. That matters for two reasons: practically, it lets organizations upgrade an existing Transformer investment into a faster, hybrid-cache model without a full retrain; scientifically, it's the first controlled, apples-to-apples comparison of SSM layer types at frontier scale, giving the field an actual data point (GKA > GDN > Mamba-2) instead of architecture folklore. It's also a vote of confidence in the broader thesis this wiki tracks across [[Attention to Mamba]] and [[Transformers Are SSMs]]: attention and SSMs are close enough, structurally, that you can move between them cheaply rather than needing to pick one at pretraining time and live with it forever.

## Related Concepts

[[Mamba]] · [[Griffin]] · [[Gated-DeltaNet-2|Gated DeltaNet-2]] · [[Attention to Mamba]] · [[Transformer]] · [[Transformers Are SSMs]] · [[Jamba]] · [[Zamba]] · [[Samba]]
