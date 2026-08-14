---
created: "2026-06-24"
title: "The Spike, the Sparse and the Sink: Anatomy of Massive Activations and Attention Sinks"
authors: "Shangwen Sun, Alfredo Canziani, Yann LeCun, Jiachen Zhu"
year: "2026"
arxiv: "2603.05498"
tags: [interpretability, attention, mechanistic-analysis, transformers]
tldr: "Massive activations and attention sinks frequently co-occur in Transformer LLMs but serve distinct purposes; their coupling is caused by the pre-norm architecture, not by any functional necessity"
citation_count: 15
---

## TL;DR

Large Transformer language models exhibit two related but distinct phenomena: **massive activations** (a small number of tokens develop extreme outlier values in a few channels, persisting as near-constant representations across all layers) and **attention sinks** (the first few tokens accumulate disproportionate attention weight, acting as a dump for attention mass that has nowhere useful to go). These two phenomena frequently co-occur, but this paper shows they serve different functions and are coupled by the pre-norm architecture — removing pre-norm causes them to decouple, establishing a causal rather than merely correlational relationship.

---

## The Problem

Prior work (notably the StreamingLLM paper, documented in [[Attention sinks]]) identified attention sinks empirically and showed that retaining the first few tokens is essential for sliding-window decoding. But *why* attention sinks form, *why* they correlate with massive activations, and whether both phenomena are necessary or architecture-contingent remained unclear.

---

## The Idea

**Functional distinction:**
- *Massive activations* operate globally across all layers, creating persistent near-constant hidden representations that function as implicit model parameters — essentially soft bias vectors baked into the activations
- *Attention sinks* operate locally to modulate attention outputs and bias certain heads toward short-range dependencies

These are different mechanisms solving different problems. Their co-occurrence is not fundamental — it is induced by the **pre-norm configuration** used in modern Transformers (LayerNorm applied before the attention/FFN sublayer rather than after). Removing pre-norm causes the two phenomena to separate: attention sinks can exist without massive activations, and vice versa.

---

## Why It Matters

- Establishes a *causal* account of why attention sinks form, rather than a purely descriptive one — the pre-norm architecture is the root cause, not some property of token content or position
- Suggests that architectural choices (norm placement) have downstream consequences for inference infrastructure: systems that pin sink tokens for cache efficiency may be working around a problem that architectural variants avoid
- Related to practical techniques like [[KV Cache Optimization]] that exploit the structural regularity of attention weights; understanding when and why that regularity holds matters for building more reliable cache eviction policies
- LeCun co-authorship makes this a notable bridge between mechanistic interpretability and the Joint Embedding Predictive Architecture (JEPA) / non-generative modeling research agenda

---

## Related Concepts

*Phenomena: [[Attention sinks]] · [[Flash Attention]] · [[Transformer]]*

*Applications: [[KV Cache Optimization]] · [[Speculative Decoding]]*
