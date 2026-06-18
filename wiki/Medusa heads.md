---
created: "2026-05-20"
title: "Medusa Heads"
tags: [glossary, medusa, speculative-decoding, inference, decoding-heads]
tldr: "K extra 2-layer MLP heads attached to the top of a frozen LLM's hidden state, each trained to predict a different future token (head k predicts token t+k). The multiple heads run in parallel during inference with zero added latency."
aliases: [Medusa heads, decoding heads, extra decoding heads]
---

## TL;DR

[[Medusa]] adds $K$ lightweight heads on top of the final hidden state $h_t$ of a frozen language model. Head $k$ (for $k = 1, \ldots, K$) predicts the token that appears $k$ positions after the current one:

$$\hat{p}_{t+k} = \text{softmax}(W_k^{(2)} \cdot \text{GELU}(W_k^{(1)} h_t))$$

All $K$ heads run simultaneously in a single forward pass — the hidden state $h_t$ is computed once, and each head applies its own two-layer MLP independently. Zero extra inference latency from the base model; only the head computations are added.

## Intuition

Think of each Medusa head as a specialist guesser: head 1 specializes in "what token comes next?", head 2 in "what comes two tokens later?", etc. These predictions are cheap (small MLPs) and run in parallel. The predictions form a tree of candidate continuations that the original model verifies in a single forward pass.

The key insight: if head $k$ correctly predicts the token at position $t+k$, the model can advance $k$ steps without running the full model $k$ times. In practice, with $K = 4$ heads, the model accepts an average of 2–3 tokens per verification pass — hence the 2.2–2.8× speedup.

Training: the heads are trained on (context, next-$K$-tokens) pairs from a dataset. Head $k$'s training target is always the token at position $t+k$. Only the head parameters are trained; the backbone LLM is frozen (Medusa-1) or jointly fine-tuned (Medusa-2).

## Why It Matters

- **They eliminate the need for a separate draft model.** A few extra MLP heads replace the complexity of managing a second model in production.
- **They're architecture-agnostic.** Any autoregressive LLM with a single hidden state output can attach Medusa heads — no architectural modification needed.
- **They're free in FLOPs relative to the base model.** The MLP heads are tiny (2 layers × ~d² parameters) vs. the full model.

## Where It Appears in This Wiki

- [[Medusa]] — the core component; K heads attached to the frozen LLM's final hidden state
- [[Tree attention]] — the Medusa heads' predictions form the tree that tree attention verifies

## Key Formula or Pseudocode

```python
# Medusa inference (simplified)
h_t = backbone(context)[:, -1, :]          # final hidden state, O(1)
drafts = [head_k(h_t).argmax(-1) for k in range(K)]  # K heads, parallel

# Build candidate tree from drafts
# One verification pass of the full model
# Accept longest matching prefix
```

## Related Concepts

[[Medusa]] · [[Tree attention]] · [[Speculative Decoding]] · [[EAGLE]] · [[Draft model]]
