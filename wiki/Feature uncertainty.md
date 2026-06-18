---
created: "2026-05-20"
title: "Feature Uncertainty"
tags: [glossary, eagle, speculative-decoding, inference, hidden-state]
tldr: "The problem EAGLE identifies: the LLM's second-to-last hidden state at step t+1 is uncertain because it depends on which token t+1 was sampled — creating a distribution over possible features rather than a single point. EAGLE resolves this by conditioning on the actual next token."
aliases: [feature uncertainty, EAGLE feature uncertainty]
---

## TL;DR

When drafting at the feature level, you want to predict $f_{t+1}$ (the next hidden state) from $f_t$ (the current hidden state). But $f_{t+1}$ depends on what token $x_{t+1}$ was sampled: different tokens lead to different hidden states at the next step. This creates uncertainty in the feature prediction — you'd need to predict an expected feature over all possible tokens, which is blurry and inaccurate.

[[EAGLE]] resolves feature uncertainty with the **one-step token advance**: the draft model receives the actual token $x_{t+1}$ (advanced by one step) as an additional input alongside $f_t$. Given both $f_t$ and $x_{t+1}$, predicting $f_{t+1}$ is now a well-determined regression problem with low uncertainty.

## Intuition

Imagine predicting where someone will be tomorrow based on today's location. If you don't know their destination, there's high uncertainty — they could go anywhere. If you know they're going to their office (the "one-step advance" equivalent), you can predict their location much more accurately.

The "one-step advance" seems circular — how do you know $x_{t+1}$? In the tree-based speculative decoding framework, the draft model works autoregressively: at position $t$, it uses the already-accepted $x_{t+1}$ to predict $\hat{f}_{t+1}$, then uses $\hat{f}_{t+1}$ to draft a distribution over $x_{t+2}$, and so on. The actual tokens are used when available; draft tokens when not.

## Why It Matters

- **It's the theoretical motivation for EAGLE's architecture.** The key insight that "feature autoregression is uncertain without the next token" leads directly to the one-step-advance design.
- **It explains why Medusa (token-level) underperforms EAGLE (feature-level).** Medusa doesn't resolve the uncertainty; EAGLE does.

## Where It Appears in This Wiki

- [[EAGLE]] — feature uncertainty is the core problem EAGLE addresses
- [[Feature-level drafting]] — the solution to feature uncertainty

## Related Concepts

[[EAGLE]] · [[Feature-level drafting]] · [[One-step token advance]] · [[Speculative Decoding]] · [[Medusa]]
