---
created: "2026-05-20"
title: "Feature-Level Drafting"
tags: [glossary, eagle, speculative-decoding, inference, hidden-state]
tldr: "Speculative decoding where the draft model predicts the LLM's internal hidden state (second-to-last layer output) rather than the next token distribution. Feature space is smoother and more predictable than token space — enabling EAGLE's 3–3.5× speedup."
aliases: [feature-level drafting, feature drafting, hidden state drafting]
---

## TL;DR

[[EAGLE]] drafts at the feature level: instead of predicting "what is the next token?", the draft model predicts "what will the second-to-last hidden state of the LLM be at the next step?". From this predicted feature, you apply the LLM's final LM head to get token probabilities. This is significantly more accurate than token-level drafting because the feature space is smooth and continuous — small changes in input produce small changes in features — while the token space is discrete and spiky.

## Intuition

Token-level drafting is hard because the next-token distribution can be multimodal and high-entropy. Even if the LLM "knows" the next word should be a noun, there might be 50 plausible nouns with similar probability. Predicting which specific noun the LLM will sample is difficult.

Feature-level drafting is easier because the hidden state is a continuous vector that changes smoothly. Given the hidden state $f_t$ at position $t$ and the actual token $x_{t+1}$ (which EAGLE knows one step later via the "one-step token advance" trick), predicting $f_{t+1}$ is a well-conditioned regression problem.

The draft model in EAGLE is a small single-layer transformer that takes $(f_t, \text{embed}(x_{t+1}))$ as input and outputs $\hat{f}_{t+1}$. The one-step advance — conditioning on the actual next token — is the key insight that removes the uncertainty from the prediction task.

## Why It Matters

- **It's the reason EAGLE outperforms Medusa.** Medusa predicts token logits directly; EAGLE predicts features. Features are more predictable, giving higher acceptance rates.
- **80–90% acceptance rate on most tasks.** Because predicting the next feature given the known next token is accurate, most drafted tokens are accepted.
- **The draft model is tiny.** A single transformer layer trained on hidden states — ~1% of the target model's parameters.

## Where It Appears in This Wiki

- [[EAGLE]] — the core innovation; draft model operates in feature space rather than token space
- [[Feature uncertainty]] — the problem that feature-level drafting resolves
- [[Speculative Decoding]] — EAGLE is a speculative decoding method with a feature-level draft model

## Related Concepts

[[EAGLE]] · [[Feature uncertainty]] · [[One-step token advance]] · [[Speculative Decoding]] · [[Medusa]] · [[Draft model]]
