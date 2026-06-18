---
created: "2026-05-20"
title: "Typical Acceptance Scheme"
tags: [glossary, medusa, speculative-decoding, inference, quality]
tldr: "Medusa's approach to token acceptance: instead of full speculative decoding rejection sampling, accept a Medusa-drafted token if the head's predicted probability exceeds a threshold — boosting acceptance rate while maintaining generation quality (bounded distribution shift)."
aliases: [typical acceptance, typical acceptance scheme, Medusa acceptance]
---

## TL;DR

Standard speculative decoding uses rejection sampling: accept a draft token iff the target model's probability is ≥ the draft model's probability. This gives exact distribution preservation but can be conservative. [[Medusa]] proposes the "typical acceptance scheme": accept a drafted token if the target model assigns it probability above a threshold $\epsilon$. This increases acceptance rate at the cost of a bounded distribution shift (the accepted tokens are "typical" for the target model, not exactly sampled from it).

## Intuition

Rejection sampling is mathematically perfect but sometimes rejects tokens that the target model would accept with reasonable probability — just not as high as the draft model's confidence. For practical applications where the exact distribution matters less than throughput, a threshold-based acceptance is simpler and achieves higher acceptance rates.

The "typical" in "typical acceptance" refers to typical sets in information theory: tokens in the typical set of the target model's distribution are accepted (anything with probability > ε), while very low-probability tokens are rejected even if the draft model is confident about them.

## Why It Matters

- **It increases acceptance rate vs. strict rejection sampling.** More tokens accepted per step = higher speedup.
- **It's simpler to implement.** No need to compute the rejection probability; just threshold the target model's logit.
- **Quality control.** The threshold ε prevents the model from accepting garbage tokens even if Medusa heads are confident.

## Where It Appears in This Wiki

- [[Medusa]] — the typical acceptance scheme is proposed as the verification method for Medusa heads

## Related Concepts

[[Medusa]] · [[Medusa heads]] · [[Speculative Decoding]] · [[EAGLE]] · [[Draft model]]
