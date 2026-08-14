---
created: "2026-07-13"
title: "Accelerating Large Language Model Decoding with Speculative Sampling"
authors: "Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, John Jumper"
year: "2023"
arxiv: "2302.01318"
tags: [inference, speculative-decoding, efficiency, foundational]
tldr: "DeepMind's independent parallel discovery of speculative decoding, termed 'speculative sampling' — demonstrated at frontier scale on Chinchilla 70B with a 4B draft model achieving ~2.6x speedup, with its own acceptance-rejection correctness proof"
citation_count: 1021
---

## TL;DR

Speculative sampling is DeepMind's independently-developed, simultaneously-published version of what [[Speculative-Decoding-Leviathan|Speculative Decoding (Leviathan 2023)]] calls speculative decoding. The core algorithm is essentially identical — draft with a small model, verify with the large model, accept/reject via modified rejection sampling with an exactness guarantee. The key contribution beyond the concurrent Leviathan et al. paper is its demonstration at frontier scale: Chinchilla 70B as the target model with a 4B draft model, achieving approximately 2.6x wallclock speedup.

## Independent Discovery

Both this paper and [[Speculative-Decoding-Leviathan|Speculative Decoding (Leviathan 2023)]] were developed independently and appeared around the same time:
- Leviathan et al. posted to arXiv November 2022 (arXiv 2211.17192) and appeared at ICML 2023 (Oral)
- Chen et al. posted to arXiv February 2023 (arXiv 2302.01318) and remained a preprint

Both provide correctness proofs for their respective acceptance/rejection schemes; both arrive at the same fundamental insight. The Leviathan et al. paper is typically cited as the canonical reference, but Chen et al. is the demonstration that the technique works at the scale researchers actually care about.

## The Idea

Identical in structure to speculative decoding: a small "draft" model (4B parameters in the paper's main experiments) generates K candidate tokens; the large target model (Chinchilla 70B) verifies all K in one parallel forward pass; tokens are accepted or rejected via a sampling scheme that provably preserves the target distribution.

The paper also introduces a useful framing: speculative sampling can be understood as importance sampling in token space — the draft model's distribution is the proposal, the target model's distribution is the target, and the accept/reject step corrects for the mismatch.

## Key Results

- ~2.6x wallclock speedup on Chinchilla 70B with a 4B draft model
- Identical output distribution to target model (proven via their acceptance/rejection correctness theorem)
- Validated at a scale more representative of production deployment than prior speculative decoding demonstrations

## Why It Matters

- Demonstrates the technique works at real frontier scale, not just on smaller experimental models
- The importance-sampling framing is a useful conceptual lens that influenced subsequent theoretical work on speculative decoding variants
- One of two canonical foundational papers for the technique, alongside [[Speculative-Decoding-Leviathan|Speculative Decoding (Leviathan 2023)]]

## Limitations

- Preprint only — never published at a major venue, unlike Leviathan et al. (ICML 2023)
- Same fundamental limitations as the Leviathan et al. approach: requires a separate draft model in memory, speedup depends on draft acceptance rate

## Related Concepts

[[Speculative-Decoding-Leviathan|Speculative Decoding (Leviathan 2023)]] · [[Medusa]] · [[EAGLE]] · [[KV Cache Optimization]] · [[Speculative Decoding]]
