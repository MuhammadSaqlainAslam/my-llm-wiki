---
created: "2026-05-20"
title: "Medusa-2"
tags: [glossary, medusa, speculative-decoding, inference, fine-tuning]
tldr: "The variant of Medusa where both the extra decoding heads and the backbone LLM are jointly fine-tuned, improving head prediction accuracy and boosting speedup to 2.3–2.8×. Introduces a small change to the output distribution relative to the original model."
aliases: [Medusa-2, Medusa 2, joint-finetuned Medusa]
---

## TL;DR

Medusa-2 jointly fine-tunes both the K [[Medusa heads]] and the backbone LLM. The additional training signal (predicting $K$ future tokens simultaneously) acts as a form of multi-token-prediction regularization, improving the backbone's representation quality. Head acceptance rates are higher than Medusa-1 because the backbone and heads are co-adapted. Speedup: 2.3–2.8×.

The tradeoff: Medusa-2 is no longer strictly lossless — the backbone's weights have changed, so the output distribution differs slightly from the original pretrained model. For most applications this is acceptable (the quality change is minimal), but for applications requiring bit-exact reproduction of a specific model's outputs, [[Medusa-1]] is safer.

## Why It Matters

- **Higher speedup than Medusa-1.** Joint training improves head accuracy, increasing the average accepted tokens per verification pass.
- **Self-distillation for cases without extra data.** When no additional training data is available, Medusa-2 can use the frozen baseline model's outputs as supervision (self-distillation).

## Where It Appears in This Wiki

- [[Medusa]] — Medusa-2 is the second training variant, offering higher speedup at the cost of slight distribution shift

## Related Concepts

[[Medusa]] · [[Medusa-1]] · [[Medusa heads]] · [[Speculative Decoding]] · [[EAGLE]]
