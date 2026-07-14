---
created: "2026-07-13"
title: "Fast Inference from Transformers via Speculative Decoding"
authors: "Yaniv Leviathan, Matan Kalman, Yossi Matias"
year: "2023"
arxiv: "2211.17192"
tags: [inference, speculative-decoding, efficiency, foundational]
tldr: "Named and formalized speculative decoding — a draft-then-verify algorithm that generates K tokens in parallel using a small draft model, accepts/rejects them via modified rejection sampling to guarantee identical output distribution to the target model, achieving 2-3x speedup at no quality cost. ICML 2023 Oral."
citation_count: 1735
---

## TL;DR

Speculative decoding generates multiple draft tokens in parallel using a small, fast draft model, then uses the large target model to verify all of them in a single forward pass. Accepted tokens are kept; rejected tokens trigger a corrected sample. The key result is an exactness proof: the output distribution is mathematically identical to sampling from the target model alone — no approximation, no quality trade-off, just speed.

## The Problem

Autoregressive decoding from large Transformer models requires one full forward pass per token. Decoding K tokens requires K serial passes, none of which can be parallelized because each token depends on all previous ones. This makes inference from large models slow regardless of hardware, since the bottleneck is memory bandwidth and sequential dependency, not raw compute.

## The Idea

Use a small, fast "draft" model to speculatively generate K candidate tokens in parallel (e.g. K=4-8 tokens ahead). Then run the large "target" model once in parallel over all K positions simultaneously — a single forward pass that produces the target model's probability distribution at each position.

Accept or reject each draft token using **modified rejection sampling**: if the target model assigns higher probability to the draft token than the draft model did, always accept. If lower, accept with probability proportional to the ratio. If rejected, sample a corrected token from the adjusted distribution. This guarantees that the final accepted tokens follow exactly the target model's distribution — no approximation.

In the best case (draft tokens all accepted), K tokens are generated for the cost of ~1 target model forward pass. In the worst case (all rejected), cost is similar to standard autoregressive decoding but no worse.

## Key Results

- 2.0-3.0x wallclock speedup demonstrated on T5-XXL and GPT-2 XL with appropriate draft models
- Identical output distribution to the target model (proven, not empirically approximated)
- No changes to model architecture or weights
- Draft model can be much smaller — even a model 10-100x smaller works well for common text

## Why It Matters

- The canonical citation for speculative decoding — introduces the name, the formalism, and the exactness proof
- Published concurrently and independently with [[Speculative-Sampling-Chen|Speculative Sampling (Chen 2023)]] — both papers converged on the same core idea; Leviathan et al. published first (Nov 2022 preprint) and provides the formal proof; Chen et al. demonstrated it at larger scale
- Every subsequent speculative decoding paper ([[Medusa]], [[EAGLE]], and others) cites this as the foundation
- Now universally deployed in production LLM serving stacks (vLLM, TensorRT-LLM, llama.cpp all implement some variant)

## Limitations

- Speedup depends entirely on draft model quality — a poor draft model (low acceptance rate) degrades to near-autoregressive speed
- Requires maintaining two models in memory simultaneously — addressed by [[Medusa]]'s decoding-head approach which eliminates the separate draft model
- Draft model must share the same tokenizer as the target model for the rejection sampling to be valid

## Related Concepts

[[Speculative-Sampling-Chen|Speculative Sampling (Chen 2023)]] · [[Medusa]] · [[EAGLE]] · [[Speculative Decoding]] · [[KV Cache Optimization]] · [[FlashAttention]]
