---
created: "2026-08-02"
title: "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"
authors: "Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang"
year: 2025
arxiv: "2503.01840"
tags: [speculative-decoding, inference, draft-model, throughput, efficiency]
aliases: [EAGLE-3, EAGLE 3]
tldr: "EAGLE's feature-prediction objective doesn't improve much with more training data — a scaling wall. EAGLE-3 abandons feature prediction for direct token prediction, and replaces top-layer-only features with multi-layer feature fusion (training-time test), letting the draft model actually benefit from data scale. Up to 6.5x speedup, ~1.4x over EAGLE-2."
theme: inference-optimization
citation_count: 0
---

# EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test

## TL;DR

[[EAGLE]] and [[EAGLE-2]] draft by predicting the target model's internal *feature* (hidden state), then reading off a token from it. That design choice turns out to be a scaling bottleneck: throwing more training data at the feature-prediction objective gives limited improvement. EAGLE-3 identifies this as the core limitation and removes it — dropping feature prediction in favor of **direct token prediction**, and replacing reliance on the top-layer feature alone with **multi-layer feature fusion** via a technique called training-time test. The draft model can now actually make use of more training data.

## The Problem

A growing trend in LLM development is scaling up *training data* to improve capability without increasing inference cost. EAGLE's draft model should benefit from the same trick — more training data, better drafts — but empirically it doesn't: scaling up EAGLE's training data gives only limited gains. The paper traces this to EAGLE's **feature prediction constraint**: predicting a specific internal hidden state is a narrower, more rigid target than predicting a token, and that rigidity caps how much a bigger training set can help.

## The Idea

Two changes, together:

1. **Abandon feature prediction for direct token prediction.** Instead of training the draft model to regress the target model's hidden state and then reading a token off it, train it to predict tokens directly. This is a less constrained objective that scales better with data.
2. **Multi-layer feature fusion (training-time test)**, replacing reliance on just the top-layer feature. Rather than conditioning only on the target model's final-layer representation, the draft model fuses information from multiple layers — giving it access to a richer signal than a single feature vector.

Together these let the draft model actually improve as training data scales up, which EAGLE-1/2's feature-prediction design didn't.

## Key Results

- Up to **6.5x speedup**, about **1.4x improvement over EAGLE-2**.
- Evaluated on both chat models and reasoning models, across five tasks.
- In the SGLang serving framework: **1.38x throughput improvement** at batch size 64.

## Why It Matters

This is a good example of diagnosing *why* an otherwise-successful method stops improving with scale, rather than just tuning it further. Recognizing that the feature-prediction objective itself was the ceiling — not insufficient training data or model capacity — let EAGLE-3 remove that ceiling directly, and the series' speedups compound as a result (EAGLE → EAGLE-2 → EAGLE-3: roughly 3x → 4x → 6.5x on comparable settings).

## Limitations

- Direct token prediction discards some of the smooth, easy-to-predict structure of feature space that motivated EAGLE-1's original design — the gain comes from better scaling, not from token prediction being intrinsically easier.
- Multi-layer feature fusion adds architectural complexity to the draft model relative to EAGLE-1/2's simpler top-layer-only design.

## Related Concepts

[[EAGLE]] · [[EAGLE-2]] · [[Speculative Decoding]] · [[Medusa]] · [[HYDRA]]
