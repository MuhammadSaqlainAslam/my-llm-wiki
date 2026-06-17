---
title: "IRIS"
aliases: ["IRIS world model", "Imagination with auto-Regression over an Inner Speech"]
year: 2022
tags: [world-models, reinforcement-learning, transformer, discrete-tokens, stub]
tldr: "A world model that tokenizes observations into discrete visual tokens (via a VQ-VAE), then uses a GPT-style autoregressive Transformer to predict the next token sequence — turning RL environment simulation into a language-modeling problem."
---

## TL;DR
IRIS encodes each game frame as a sequence of discrete latent tokens (like codebook indices from a VQ-VAE), then trains a causal Transformer to predict the next frame's tokens given the current frame's tokens and action. The policy sees these predicted token sequences as its world model rollouts. Result: Atari performance using only 2 hours of real gameplay.

## Intuition
If you can compress a game frame into ~16 discrete tokens, then "predicting the next frame" becomes "predicting 16 tokens given 16 + action" — exactly the language modeling setup where Transformers excel. You get sample efficiency from the world model (practice in imagination) and model quality from the autoregressive prior over tokens.

## Why It Matters
- Showed that the GPT/LLM training recipe transfers almost directly to world modeling
- Achieves superhuman Atari scores with very limited real-environment interaction
- Influential baseline for follow-up world models including [[DIAMOND]] and [[Looped World Models|LoopWM]]

## See Also
[[DreamerV3]] · [[DIAMOND]] · [[Looped World Models]] · [[Recurrent State Space Model]]
