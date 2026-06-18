---
created: "2026-05-20"
title: "One-Step Token Advance"
tags: [glossary, eagle, speculative-decoding, inference]
tldr: "EAGLE's key trick: the draft model receives the actual next token embedding (shifted one step forward) as input, resolving the uncertainty in feature-level prediction. Predicting f_{t+1} given both f_t and x_{t+1} is vastly more accurate than predicting from f_t alone."
aliases: [one-step token advance, token advance, shifted token input]
---

## TL;DR

In [[EAGLE]], the feature draft model takes two inputs: the current feature $f_t$ (second-to-last hidden state) and the embedding of the *next* token $\text{embed}(x_{t+1})$. The token $x_{t+1}$ is "advanced by one step" relative to what you're predicting ($f_{t+1}$). This conditioning resolves [[Feature uncertainty]]: given both the current context and the actual next token, predicting the next hidden state becomes a well-posed regression.

## Intuition

This sounds circular, but it's not. The draft model predicts *feature $t+1$* given *token $t+1$* — that is, it's predicting what the hidden state will look like *after* processing token $t+1$. You know token $t+1$ because you're building a tree of candidates: for each candidate token in the tree, you run the draft model with that token as input to predict the next feature.

In the tree structure:
- Root: actual hidden state $f_t$ and actual token $x_t$ (from the base model)
- Level 1 drafts: for each candidate $x_{t+1}^{(i)}$, predict $\hat{f}_{t+1}^{(i)}$ from $(f_t, x_{t+1}^{(i)})$
- Level 2 drafts: for each $(x_{t+1}^{(i)}, \hat{f}_{t+1}^{(i)})$, sample $x_{t+2}^{(j)}$ from LM head, then predict $\hat{f}_{t+2}^{(ij)}$, etc.

The key: the draft model is given the actual candidate token as a *fact*, not something it needs to guess. This makes feature prediction accurate (80–90% acceptance).

## Why It Matters

- **It's the single design decision that makes EAGLE significantly better than Medusa.** Conditioning on the next token resolves the main source of prediction error.
- **It's computationally cheap.** Adding $\text{embed}(x_{t+1})$ to the draft model input costs negligible computation.

## Where It Appears in This Wiki

- [[EAGLE]] — the one-step advance is the architectural centerpiece
- [[Feature-level drafting]] — the overall approach that the one-step advance enables
- [[Feature uncertainty]] — the problem that the one-step advance resolves

## Related Concepts

[[EAGLE]] · [[Feature-level drafting]] · [[Feature uncertainty]] · [[Speculative Decoding]] · [[Tree attention]]
