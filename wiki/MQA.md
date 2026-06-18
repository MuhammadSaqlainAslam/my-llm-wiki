---
created: "2026-05-20"
title: "MQA (Multi-Query Attention)"
tags: [glossary, attention, efficiency, kv-cache, inference]
tldr: "An extreme KV cache reduction: all query heads share a single key-value head. Reduces KV cache size by the number of heads (e.g., 32×) with some quality degradation — the precursor to GQA's more balanced approach."
aliases: [MQA, Multi-Query Attention, multi-query attention]
---

## TL;DR

Multi-Query Attention (Shazeer, 2019) replaces the $h$ key and value heads in [[Multi-head attention]] with a single shared KV head. All query heads attend to the same K, V projections. This reduces the [[KV Cache]] by factor $h$ (e.g., 32× for GPT-style models) at the cost of some model quality — each query head loses the ability to learn head-specific key/value patterns.

[[GQA]] (Grouped Query Attention) is the middle ground: $h/g$ KV heads shared across groups of $g$ query heads. LLaMA 2 uses GQA with 8 KV heads vs. 64 query heads — 8× cache reduction with near-zero quality loss, which is why GQA replaced MQA in most production models.

## Where It Appears in This Wiki

- [[Multi-head attention]] — MQA is a variant reducing the KV head count to 1
- [[GQA]] — GQA generalizes MQA; uses $h/g$ KV heads instead of 1

## Related Concepts

[[GQA]] · [[Multi-head attention]] · [[KV Cache]] · [[LLaMA 2]] · [[Flash Attention]]
