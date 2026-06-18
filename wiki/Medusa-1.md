---
created: "2026-05-20"
title: "Medusa-1"
tags: [glossary, medusa, speculative-decoding, inference, fine-tuning]
tldr: "The variant of Medusa where only the extra decoding heads are fine-tuned, while the backbone LLM is frozen. Guarantees lossless inference — identical output distribution to the original model — at 2.2× speedup."
aliases: [Medusa-1, Medusa 1, frozen-backbone Medusa]
---

## TL;DR

Medusa-1 fine-tunes only the K extra [[Medusa heads]] on a dataset of (context, next-K-tokens) pairs. The backbone LLM's weights are frozen throughout training. Because the backbone is unchanged, Medusa-1 guarantees **lossless inference**: the accepted tokens follow exactly the original model's distribution (via speculative decoding's rejection sampling mechanism). Speedup: ~2.2× with no quality loss.

## Why It Matters

- **Zero-risk deployment.** Medusa-1 is a drop-in speedup for any pretrained model — quality guaranteed identical to the unmodified model.
- **No retraining of the backbone.** Only the small MLP heads need training, which is fast (hours on a single GPU).

## Where It Appears in This Wiki

- [[Medusa]] — Medusa-1 is one of two training variants

## Related Concepts

[[Medusa]] · [[Medusa-2]] · [[Medusa heads]] · [[Speculative Decoding]] · [[Tree attention]]
