---
created: "2026-06-03"
title: "Positional Encoding"
tags: [glossary, transformers]
tldr: "Injects token order information into embeddings since self-attention is inherently permutation-invariant."
---

## TL;DR
Transformers have no built-in sense of sequence order — positional encoding adds that signal, either via fixed sinusoids or learned vectors, so token 42 "knows" it's not token 1.

## Intuition
Self-attention computes pairwise dot products over a *set*, not a sequence. Shuffle the input tokens and you get identical attention scores — catastrophic for language. Positional encoding fixes this by adding a position-dependent vector to each token embedding before any attention happens. The original "Attention Is All You Need" used sinusoids of varying frequencies (wavelengths from 2π to 10000·2π), giving each position a unique fingerprint that generalizes to unseen lengths.

The key tension: absolute encodings (original sine/cosine, learned) bake in a fixed context window and generalize poorly beyond it. Relative encodings like [[RoPE]] instead rotate the query/key vectors by an angle proportional to their *relative* distance, which naturally handles extrapolation and is now the dominant approach — used in LLaMA, Mistral, and most modern LLMs with context windows stretching to 128K+ tokens.

## Why It Matters
- **Context length scaling**: Poor positional encoding is the primary bottleneck to extending context; techniques like YaRN and RoPE interpolation allow stretching a 4K model to 128K without full retraining.
- **Attention pattern quality**: Without positional signal, the model cannot distinguish "dog bites man" from "man bites dog" — positional encoding directly determines what syntactic and semantic structure the model can learn.
- **[[KV Cache]] efficiency**: Relative position schemes like [[RoPE]] encode position at attention time, not embedding time, meaning cached keys/values remain valid as the sequence grows — critical for efficient autoregressive decoding.

## Related Concepts
[[RoPE]], [[Transformer]], [[Attention]], [[KV Cache]], [[FlashAttention]]
