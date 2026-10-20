---
created: "2026-10-19"
title: "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models"
authors: "David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, Adam Santoro (Google DeepMind)"
year: 2024
arxiv: "2404.02258"
citation_count: 0
tags: [architecture, efficiency, routing, moe, adaptive-compute, scaling, inference]
tldr: "Stop spending the same FLOPs on every token at every layer. Cap each block at top-k tokens (fixed k → static graph and known tensor sizes); the router picks *which* tokens go through attention+MLP. Matches dense baselines at equal FLOPs/wall-clock train, but far fewer FLOPs per forward and up to 50% faster sampling steps."
aliases: [Mixture-of-Depths, MoD, dynamically allocating compute]
theme: efficiency
---

# Mixture-of-Depths (MoD)

> David Raposo, et al. (Google DeepMind), "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models", 2024 (arXiv:2404.02258)

## The Problem / Motivation

A Transformer block runs **every** token through attention + MLP. Easy tokens (syntax glue) and hard tokens (new entity, arithmetic step) get the same FLOPs.

Compare [[Mixture-of-Experts]]: MoE varies **where** (which FFN) at constant **width** of compute. MoD varies **whether / how many tokens** enter a block — **compute along depth/time**, not expert identity.

Constraint for real systems: conditional compute must not break **static shapes** (compiler/kernel friendliness).

## The Idea

**Cap** the number of tokens that participate in a layer’s expensive path at **$k$** (fixed a priori). A router ranks tokens; only **top-$k$** run that block; the rest skip (residual continues). $k$ fixed ⇒ **static computation graph**, known tensor sizes; **which** tokens are fluid ⇒ dynamic, context-sensitive FLOPs.

## Architecture / Method

```
tokens ──► router scores per block
              │
        top-k selected ──► self-attn + MLP (full block)
        others     ──► skip (identity / residual only)
```

- **Fixed budget per block:** always exactly $k$ “active” tokens (or a fixed capacity factor).
- **Learned routing:** network decides *who* needs depth at *this* step.
- **Across layers:** different positions can be experts at different depths — non-uniform in **time and depth**.
- Predictable **total** FLOPs; dynamic **allocation**.

## Key Results

| Metric | Result |
|---|---|
| Quality at matched FLOPs / wall-clock **train** | Matches **dense baselines** |
| FLOPs **per forward** | **A fraction** of dense |
| Post-training **sampling steps** | Up to **~50% faster** |
| Graph | **Static** shapes (unlike many dynamic conditional-compute methods) |

## Comparison to Prior Work

- vs **[[Mixture-of-Experts]] / [[Mixtral]]** — MoE routes to parameters; MoD routes **whether to compute** at a block (orthogonal; can stack).
- vs **early exit / [[LayerSkip]]** — early exit ends the **whole** depth for a token; MoD is **per-block** top-k among tokens (finer).
- vs **[[TokenFormer]]** — change projection parameterization; MoD keeps the block and **subsets tokens**.
- vs **[[Native Sparse Attention|NSA]]** — sparsify **attention’s KV**; MoD sparsifies **which tokens enter the block**.

## Limitations

- Router needs care (collapse to the same easy tokens / load-balance across positions).
- Skipping blocks is a capacity tradeoff on some hard tokens the router underestimates.
- Reported gains are DeepMind-scale research models; kernels must honor the top-k gather/scatter.
- Not a free win on every workload (short chat vs long reasoning differ).
- Interaction with MoE, TP/SP ([[Nemotron-3]] style) is extra design work.

## Why It Matters

This wiki keeps asking *how do you spend fewer bytes/FLOPs per token of intelligence*. MoD is the clean **adaptive depth / per-token FLOPs** answer with a **static graph** — the systems-friendly kind. It sits beside [[Mixture-of-Experts]] (parameter sparsity), [[LayerSkip]] (self-speculative depth), and [[BitNet b1.58]] (cheaper multiplies) as the third knob: **skip work for easy tokens**.

## Related Concepts

[[Mixture-of-Experts]] · [[Mixtral]] · [[LatentMoE]] · [[Routing-Free-MoE]] · [[Load Balancing Loss]] · [[LayerSkip]] · [[TokenFormer]] · [[Native Sparse Attention]] · [[BitNet b1.58]] · [[Inference-Time-Scaling]] · [[Recurrent-Depth-Reasoning]] · [[Transformer]] · [[GPU]]