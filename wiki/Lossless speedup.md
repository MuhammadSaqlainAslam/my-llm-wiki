---
created: "2026-05-20"
title: "Lossless Speedup"
tags: [glossary, speculative-decoding, inference, medusa, eagle, quality]
tldr: "A speedup method is 'lossless' if the accelerated decoding produces tokens from exactly the same distribution as standard autoregressive decoding. Achieved via speculative decoding's rejection sampling; EAGLE is lossless by this definition at 3–3.5×."
aliases: [lossless speedup, lossless speculative decoding, lossless acceleration]
---

## TL;DR

A speculative decoding method is **lossless** if the accepted tokens follow the same probability distribution as the original model would produce via standard autoregressive sampling. Rejection sampling guarantees this: the draft token $x$ at position $t$ is accepted with probability $\min(1, p_\text{target}(x) / p_\text{draft}(x))$, and if rejected, a correction sample is drawn from the target model. The result: regardless of what the draft model proposes, the final output distribution is exactly the target model's distribution.

## Intuition

Consider two ways to roll a die:
1. Roll directly (target model)
2. Propose a roll from a loaded die (draft), then correct with rejection sampling

Rejection sampling makes option 2 equivalent to option 1 in distribution — even though you're using a different sampler. The correction step is the key.

[[EAGLE]], [[Medusa-1]], and standard [[Speculative Decoding]] all use rejection sampling and are therefore lossless. [[Medusa-2]] is slightly lossy because the backbone is fine-tuned. [[Typical acceptance scheme|Typical acceptance]] is lossy but bounded (accepted tokens are within the typical set).

## Why It Matters

- **It's the quality guarantee for production deployment.** If your speedup is lossless, you don't need to re-evaluate the model — quality is identical to the base model by construction.
- **It's the standard benchmark for inference speedup methods.** Reporting "lossless 3× speedup" (as EAGLE does) is a strong claim: same quality, three times faster.

## Where It Appears in This Wiki

- [[EAGLE]] — EAGLE is lossless via rejection sampling; 3–3.5× speedup
- [[Medusa]] — Medusa-1 is lossless (frozen backbone); Medusa-2 is slightly lossy
- [[Speculative Decoding]] — the general framework; lossless when using rejection sampling

## Related Concepts

[[Speculative Decoding]] · [[EAGLE]] · [[Medusa]] · [[Medusa-1]] · [[Draft model]] · [[Typical acceptance scheme]]
