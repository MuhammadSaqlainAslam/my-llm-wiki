---
created: "2026-05-12"
title: "Verification Step"
tags: [speculative-decoding, inference, sampling]
tldr: "The single parallel forward pass through the big \"target\" model in [[Speculative Decoding]] that scores all $K$ draft tokens at once, accepts the longest matching prefix, and rejects the rest. Equivalent to autoregressive sampling in distribution — no quality loss."
---

# Verification Step

Given $K$ draft tokens proposed by the [[Draft model]], the verifier runs **one** forward pass with all $K$ tokens visible at once (causal-mask attention, just like training). It produces $K$ probability distributions — one per position — each conditioned on the actual prompt + the previously accepted draft tokens. For each position $k$, the verifier accepts the draft if $\min(1, p_\text{verifier}(t_k) / p_\text{draft}(t_k)) \ge u$ where $u \sim \text{Uniform}(0, 1)$ — a rejection-sampling rule that's provably equivalent in distribution to direct sampling from the verifier. The first rejection ends the accepted prefix; the verifier samples its own token at that position, and the loop restarts. The whole verification pass costs roughly the same wall-clock as one normal autoregressive step — that's why it's a win.

## Where it appears

- [[Speculative Decoding]]

---

*Related: [[Draft model]] · [[Multi-Token Prediction]] · [[KV Cache]]*
