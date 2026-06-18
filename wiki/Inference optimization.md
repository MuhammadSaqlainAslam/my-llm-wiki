---
created: "2026-05-20"
title: "Inference Optimization"
tags: [glossary, inference, throughput, latency, efficiency]
tldr: "The family of techniques that reduce the wall-clock time, memory, and cost of running a trained LLM — including speculative decoding, KV cache compression, quantization, batching strategies, and hardware-software co-design."
aliases: [inference optimization, LLM inference optimization, inference efficiency]
---

## TL;DR

Inference optimization covers everything that happens after a model is trained: how to serve it faster, cheaper, and at larger scale. The key insight from [[Hardware Acceleration for Neural Networks]]: LLM inference is **memory-bandwidth bound**, not compute-bound. Each generated token requires reading the entire model from HBM. Optimizations target:

1. **Reduce memory reads**: [[Speculative Decoding]] generates $K$ tokens per model read; [[KV Cache Optimization]] reduces the cache that must be read per step
2. **Reduce the model size**: quantization (FP8, INT8, NVFP4) shrinks the model proportionally
3. **Increase arithmetic intensity**: larger batches amortize the model load across more tokens
4. **Overlap compute with memory**: speculative decoding, continuous batching, async prefill

The three main categories in this wiki:
- **Speculative decoding**: [[Speculative Decoding]], [[Medusa]], [[EAGLE]]
- **KV cache management**: [[KV Cache Optimization]], [[GQA]], [[Sliding window attention]]
- **Hardware-aware kernels**: [[Flash Attention]], [[Hardware-Aware Scan]]

## Why It Matters

- **Inference cost dominates training cost over a model's lifetime.** A model trained once is served billions of times. Optimizing inference by 2× is worth more than training 2× faster.
- **It enables deployment at scale.** Without inference optimization, serving 70B models at reasonable cost is infeasible.

## Where It Appears in This Wiki

- [[EAGLE]] — references inference optimization as the broader context
- [[Medusa]] — references inference optimization as the broader context

## Related Concepts

[[Speculative Decoding]] · [[KV Cache Optimization]] · [[Flash Attention]] · [[EAGLE]] · [[Medusa]] · [[Hardware Acceleration for Neural Networks]]
