---
created: "2026-05-20"
title: "Draft-Model-Free Speculative Decoding"
tags: [glossary, speculative-decoding, medusa, inference, efficiency]
tldr: "Speculative decoding methods that generate draft tokens without a separate smaller model — using extra heads (Medusa), Jacobi iteration (Lookahead), or other self-contained mechanisms. Eliminates the deployment complexity of managing a draft model."
aliases: [draft-model-free, model-free speculative decoding, self-speculation]
---

## TL;DR

Standard [[Speculative Decoding]] requires a separate draft model (typically a smaller version of the target model) to propose candidate tokens. Draft-model-free methods achieve the same speedup pattern without this second model, by embedding the drafting mechanism inside or alongside the target model.

The main approaches:
- **[[Medusa]]**: K extra MLP heads on the frozen backbone predict K future tokens
- **[[Lookahead Decoding]]**: Jacobi iteration with no extra parameters
- **Multi-Token Prediction (MTP) heads**: auxiliary heads trained with the backbone (used in DeepSeek, Nemotron-3)

## Why It Matters

- **Operational simplicity.** No need to version-match a draft model to the target model, host two models, or worry about draft model size/quality tradeoffs.
- **Zero memory overhead for the draft model.** Medusa heads are tiny; Lookahead uses no extra parameters.

## Where It Appears in This Wiki

- [[Medusa]] — the primary draft-model-free method; extra heads replace the draft model

## Related Concepts

[[Medusa]] · [[EAGLE]] · [[Speculative Decoding]] · [[Lookahead Decoding]] · [[Multi-Token Prediction]]
