---
created: "2026-06-11"
title: "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization"
authors: "Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, Amir Gholami"
year: "2024"
arxiv: "2401.18079"
aliases: ["KVQuant"]
tags: [kv-cache, quantization, inference, long-context]
tldr: "Per-channel and per-token non-uniform quantization of the KV cache, calibrated to its heavy-tailed activation distribution — achieves under 3-bit KV cache with minimal perplexity degradation, enabling million-token-scale context on a single GPU. NeurIPS 2024."
citation_count: 600
---

## TL;DR

KV cache activations have heavy tails — a few outlier channels are much larger than the rest, which breaks naive uniform quantization. KVQuant analyzes these activation patterns and applies non-uniform quantization calibrated per-channel (for keys) and per-token (for values), achieving under 3-bit KV cache quantization with under 0.1 perplexity degradation on LLaMA models, enabling inference with context lengths up to ~10 million tokens on a single GPU. NeurIPS 2024.

## The Problem

As context length, batch size, or model size grows, the KV cache becomes the dominant contributor to GPU memory usage and the main bottleneck for inference latency and throughput. Naive uniform quantization of the KV cache to low bit-widths degrades model quality badly, because KV activations have a heavy-tailed distribution — a small number of outlier channels/tokens carry disproportionate magnitude, and a uniform quantization grid wastes precision on the bulk of values while clipping or poorly representing the outliers.

## The Idea

KVQuant performs a detailed empirical analysis of where the outliers in KV cache activations live, finding that key activations have outlier structure concentrated in specific channels (consistent across tokens), while value activations don't share this per-channel structure. This motivates an asymmetric quantization scheme: **per-channel quantization for keys** (a separate scale/zero-point per channel, capturing the channel-consistent outlier structure) and **per-token quantization for values** (a separate scale per token). Additional refinements include non-uniform quantization grids (calibrated to the actual activation distribution rather than assuming uniform spacing) and handling a small number of numerical outliers with higher precision separately from the bulk of quantized values.

## Key Results

- Under 3-bit KV cache quantization with under 0.1 perplexity degradation on LLaMA, LLaMA-2, and Mistral models
- Enables inference with up to ~10 million token context length on a single GPU by combining the quantized KV cache with existing serving infrastructure
- Substantial memory reduction directly translates to either longer context or higher batch throughput at fixed memory budget

## Why It Matters

- One of the most detailed empirical analyses of *why* KV cache activations are hard to quantize (channel vs. token outlier structure), not just a quantization scheme applied blindly
- The asymmetric per-channel/per-token design is a template that subsequent KV cache quantization work builds on
- Directly enables the "very long context" regime that later work like [[MagicDec]] analyzes the serving economics of

## Limitations

- The calibration step (determining per-channel/per-token quantization parameters) adds a preprocessing cost and requires representative calibration data
- Handling numerical outliers with separate higher-precision storage adds implementation complexity relative to a naive uniform quantization scheme

## Related Concepts

[[KV Cache]] · [[NVFP4]] · [[Quantization]] · [[GQA]] · [[MagicDec]] · [[Speculative Decoding]]
