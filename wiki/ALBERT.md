---
title: "ALBERT"
aliases: ["ALBERT", "A Lite BERT"]
year: 2019
tags: [bert, parameter-sharing, factorization, pretraining, nlp, stub]
tldr: "Parameter-efficient BERT variant that uses cross-layer weight sharing (all encoder layers share one parameter set) and factorized embedding parameterization — achieving similar performance to BERT-large with 18× fewer parameters, making it an early looped-transformer proof of concept."
---

## TL;DR
BERT stacks 24 separate transformer layers. ALBERT asks: what if all 24 layers were *the same* layer, repeated 24 times? It also factorizes the embedding matrix into a small lookup table × a projection, saving parameters at the vocabulary dimension. With these two tricks, ALBERT-xxlarge matches or beats BERT-large on GLUE/SQuAD with far fewer parameters (though similar FLOPs — weight sharing saves storage, not compute).

## Intuition
ALBERT is one of the earliest empirical validations that layer weight sharing doesn't kill performance in Transformers. The same parameters can be applied repeatedly to refine a hidden state — a key insight that motivates [[Universal Transformer]], [[Looped World Models|LoopWM]], and [[Fixed-Point Reasoners Stable and Adaptive Deep Looped Transformers|FPRM]].

## Why It Matters
- Early proof that parameter-shared (looped) Transformers can match individually-parameterized ones
- Separates *parameter count* from *effective depth* — a theme revisited by every looped architecture paper since
- Used as a comparison baseline for looped world models and reasoning models

## See Also
[[Universal Transformer]] · [[Looped World Models]] · [[Adaptive Computation Time]] · [[Deep Equilibrium Models]] · [[Attention Is All You Need]]
