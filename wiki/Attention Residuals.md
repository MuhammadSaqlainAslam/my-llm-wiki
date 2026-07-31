---
created: "2026-07-30"
title: "Attention Residuals"
authors: "Kimi Team, Guangyu Chen, Yu Zhang, Jianlin Su, et al."
year: 2026
tags: [architecture, attention, residual-connections, moe, kimi]
aliases: [AttnRes, Attention Residuals, Block AttnRes]
tldr: "Replace fixed-weight residual accumulation (PreNorm's sum of all prior layer outputs) with softmax attention over preceding layers, so each layer learns input-dependent weights for how much of each earlier layer to keep. Block AttnRes makes this affordable at scale; integrated into a 48B/3B-activated Kimi Linear model pretrained on 1.4T tokens with gains on every evaluated task."
theme: foundational
arxiv: "2603.15031"
citation_count: 40
---

# Attention Residuals

## TL;DR

PreNorm residual connections — standard in modern LLMs — accumulate every layer's output with a fixed weight of 1. That uniform accumulation causes hidden-state magnitude to grow unboundedly with depth, diluting the relative contribution of any single layer the deeper the network gets. **Attention Residuals (AttnRes)** replaces the fixed sum with softmax attention over the preceding layers' outputs, so each layer learns, per-input, how much of each earlier layer's representation to keep. A block-wise variant (**Block AttnRes**) makes the extra memory/communication cost practical at large scale.

## The Problem

A residual stream is built by adding each layer's output to a running total: $h_l = h_{l-1} + f_l(h_{l-1})$. Every term gets weight 1, regardless of how relevant that layer's output still is by the time you're 80 layers deep. This causes:

- **Uncontrolled hidden-state growth** — the residual stream's magnitude increases with depth because nothing ever gets down-weighted.
- **Progressive dilution** — as the stream grows, each individual layer's contribution becomes a smaller and smaller fraction of the total, so deep layers struggle to influence the final representation as strongly as shallow ones.

## The Idea

Instead of summing all prior layer outputs with fixed unit weight, let the model decide the weights — with **softmax attention over preceding layer outputs**. Each layer now selectively aggregates earlier representations using learned, input-dependent weights, rather than blindly keeping everything at full strength forever.

The direct version of this — attending over *every* preceding layer's full output — is expensive: it multiplies the memory and communication cost of what used to be a single addition. **Block AttnRes** fixes this by partitioning the network's layers into blocks and attending over block-level representations instead of per-layer ones, cutting the memory footprint while preserving most of the accuracy gains. Combined with cache-based pipeline communication and a two-phase computation strategy, Block AttnRes ends up being close to a drop-in replacement for a standard residual connection.

## Key Results

- Scaling-law experiments show the improvement holds consistently across model sizes.
- Ablations confirm the gain specifically comes from *content-dependent, depth-wise* selection (not just any richer connection pattern).
- Integrated into the **Kimi Linear** architecture (48B total / 3B activated parameters) and pretrained on 1.4T tokens: AttnRes mitigates PreNorm's dilution effect, produces more uniform output magnitudes and gradient distributions across depth, and improves downstream performance on every evaluated task.

## Why It Matters

Residual connections have been essentially unchanged since ResNet — a fixed-weight sum baked into every Transformer since the original architecture. This is evidence that even that basic wiring can be improved with the same content-dependent selection trick ([[Linear attention|attention itself]]) that already displaced fixed convolution kernels and fixed recurrence. Since Block AttnRes is a near-drop-in replacement, it's a relatively low-risk architectural upgrade for anyone already training large MoE/hybrid models.

## Limitations

- Full (non-blocked) AttnRes is too memory/communication-heavy for large-scale training — Block AttnRes trades some of the accuracy gain back for practicality.
- Validated within one model family (Kimi Linear) at one scale (48B/3B); generality across very different architectures isn't yet established.

## Related Concepts

[[Transformer]] · [[Kimi-K2|Kimi K2]] · [[Kimi-K3|Kimi K3]] · [[Linear attention]] · [[Mixture-of-Experts]]

**Where it appears:** integrated into the 48B/3B Kimi Linear model above; also adopted directly in **[[Kimi-K3|Kimi K3]]** (2.8T/104B activated), alongside Kimi Delta Attention, for depth-wise information flow.
