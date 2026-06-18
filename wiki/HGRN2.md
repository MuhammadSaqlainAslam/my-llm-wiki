---
created: "2026-06-03"
title: "HGRN2"
tags: [glossary, recurrent-models]
tldr: "An improved linear recurrent architecture that adds outer-product state expansion to HGRN, achieving competitive performance with Transformers while maintaining O(1) memory at inference."
---

## TL;DR
HGRN2 upgrades the original Hierarchical Gated Recurrent Network by expanding its hidden state via outer products, giving it expressivity closer to SSMs like Mamba without the hardware complexity.

## Intuition
The core problem with vanilla RNNs is their hidden state is too small and scalar-gated — you lose information fast. HGRN2 fixes this by constructing a larger effective state through outer products of two vectors (think rank-1 expansion), letting the model remember more per recurrence step without ballooning parameters. The recurrence still runs in O(1) memory per token at inference, but now the state is rich enough to actually model long-range dependencies that earlier RNNs fumbled.

Concretely, on language modeling benchmarks HGRN2 with ~3B parameters matches or beats similarly sized Transformers on standard tasks like Wikitext-103, while being significantly faster at inference due to no KV cache growth. The training can be parallelized via a parallel prefix scan (like S4/Mamba), so you don't sacrifice throughput during training either.

## Why It Matters
- **Inference efficiency**: Fixed-size recurrent state means memory stays flat regardless of sequence length — critical for long-context deployment where [[KV Cache]] costs explode.
- **SSM-competitive without selective scan complexity**: Achieves Mamba-class modeling power through simpler outer-product state expansion, making it easier to implement and optimize on standard hardware.
- **Viable Transformer alternative at scale**: Demonstrates that linear recurrent models can close the gap on LLM benchmarks at billion-parameter scale, validating the non-attention path forward.

## Related Concepts
[[State Space Model]] [[Mamba]] [[Transformer]] [[Attention]] [[FlashAttention]]
