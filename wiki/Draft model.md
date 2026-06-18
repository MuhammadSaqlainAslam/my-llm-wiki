---
created: "2026-05-12"
title: "Draft Model"
tags: [speculative-decoding, inference, throughput]
tldr: "The small, fast model in [[Speculative Decoding]] that proposes $K$ candidate tokens for the larger \"target\" model to verify in parallel. Can be a separately trained small model, an early-exit head, or a [[Multi-Token Prediction]] auxiliary head."
---

# Draft Model

Speculative decoding amortizes one forward pass of the big model across multiple generated tokens by having a *fast* draft model propose them first. The draft model can be: (i) a small standalone LM trained to mimic the target's distribution, (ii) an early-exit head that takes the target's intermediate-layer representations and predicts a few steps ahead (EAGLE, EAGLE-3), or (iii) the target model's own [[Multi-Token Prediction]] heads — zero extra parameters because the heads exist for free as a training byproduct. The draft has to be *fast enough* that drafting + verifying $K$ tokens costs less wall-clock than $K$ direct verifier passes; that constraint is what limits how large the draft model can be.

## Where it appears

- [[Speculative Decoding]]
- [[Nemotron-3]]

---

*Related: [[Multi-Token Prediction]] · [[Verification step]]*
