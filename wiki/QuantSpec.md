---
created: "2026-06-11"
title: "QuantSpec: Self-Speculative Decoding with Hierarchical Quantized KV Cache"
authors: "Rishabh Tiwari, Haocheng Xi, Aditya Tomar, Coleman Hooper, Sehoon Kim, Maxwell Horton, Mahyar Najibi, Michael W. Mahoney, Kurt Keutzer, Amir Gholami"
year: "2025"
arxiv: "2502.10424"
aliases: ["QuantSpec"]
tags: [speculative-decoding, quantization, kv-cache, inference]
tldr: "Self-speculative decoding where the draft model shares the target model's architecture but uses a hierarchical 4-bit quantized KV cache and weights — no separate draft model to maintain, >90% acceptance rate, ~2.5x end-to-end speedup. ICML 2025."
citation_count: 25
---

## TL;DR

QuantSpec is a self-speculative decoding method: instead of a separate, smaller draft model (as in [[Speculative Decoding]]), the draft uses the *same* model architecture as the target, but runs with a hierarchical 4-bit quantized KV cache (and quantized weights) instead of the target's full-precision cache. Because the draft shares the target's architecture and weights, its predictions align closely with the target's distribution, giving high acceptance rates (>90%) while eliminating the deployment complexity of maintaining a second, separate draft model. ICML 2025.

## The Problem

Standard speculative decoding requires deploying and maintaining a separate draft model alongside the target model — extra memory footprint, an extra model to keep in sync as the target is updated, and often a nontrivial engineering effort to find or train a draft model with acceptable acceptance rates against the target. Self-speculative approaches (using the target model itself, or a heavily-modified variant of it, as the draft) avoid this but need some way to make each self-speculative step cheap enough to be worth running.

## The Idea

QuantSpec avoids the separate-draft-model problem by drafting with a quantized version of the target model itself: a hierarchical 4-bit quantization scheme is applied to both the KV cache and the model weights during the draft pass, making each draft step significantly cheaper than a full-precision forward pass, while the verification pass runs the target model at full precision as usual. Because the draft is architecturally identical to the target (same weights, just quantized), the draft's output distribution tracks the target's far more closely than an independently-trained smaller draft model would, yielding a high acceptance rate.

## Key Results

- Acceptance rates above 90% due to architectural identity between draft and target
- ~2.5x end-to-end speedup, consistent across evaluated settings
- No separate draft model to train, store, or maintain — only a quantized view of the existing target model

## Why It Matters

- Removes one of the standing limitations of the [[Speculative-Decoding-Leviathan|Speculative Decoding (Leviathan 2023)]] / [[Speculative-Sampling-Chen|Speculative Sampling (Chen 2023)]] line of work — needing a separate draft model — in a different way than [[Medusa]]'s extra-heads approach: quantization instead of extra parameters
- Connects the speculative decoding cluster directly to the KV-cache-quantization cluster ([[KVQuant]])
- Practical for deployments that already have quantized-inference infrastructure, since the same quantization machinery serves both the draft path and (optionally) general inference

## Limitations

- Acceptance rate and speedup depend on how well the quantized draft preserves the target's distribution — more aggressive quantization would likely trade acceptance rate for cheaper draft steps
- Since the draft is a quantized version of the same weights, it doesn't benefit from a truly independent, differently-specialized draft model in the way a small separately-trained draft model might for certain domains

## Related Concepts

[[Speculative Decoding]] · [[Quantization]] · [[KV Cache]] · [[Multi-Token Prediction]] · [[KVQuant]] · [[Medusa]]
