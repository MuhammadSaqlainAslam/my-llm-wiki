---
created: "2026-06-03"
title: "Hawk"
tags: [glossary, architecture, recurrent-models]
tldr: "A linear recurrent architecture from DeepMind's Griffin paper that matches Transformer quality while using O(1) memory per token at inference time."
---

## TL;DR
Hawk is a pure recurrent sequence model using Real-Gated Linear Recurrent Units (RG-LRU) that achieves Transformer-competitive perplexity with constant inference-time memory, trained on up to 300B tokens.

## Intuition
Hawk replaces attention entirely with a gated linear recurrence. Each layer maintains a fixed-size hidden state (no growing KV cache) — think of it as a learned, data-dependent exponential moving average over the sequence. The key innovation is the RG-LRU cell: it uses an input gate and a learned per-channel decay rate `a ∈ (0,1)`, initialized so the effective recurrence timescale covers ~1K–2K tokens. This gives it memory, but bounded memory.

At 7B parameters trained on 300B tokens, Hawk matches or beats similarly-sized Transformers on downstream tasks while using constant memory per decoding step — no KV cache that grows with context length. Throughput gains become dramatic at long sequences: where Transformer inference memory scales as O(n·layers·d_head), Hawk stays flat. The tradeoff is hard recall over very long contexts — anything the recurrent state "forgot" is gone.

## Why It Matters
- **Inference efficiency**: Eliminates the KV cache entirely, making long-context generation practical on memory-constrained hardware — critical for edge deployment or high-throughput serving.
- **Training parity with Transformers**: Unlike earlier SSMs, Hawk closes the quality gap on language modeling benchmarks, suggesting recurrent models are no longer a second-class citizen at scale.
- **Hybrid design signal**: Hawk's sibling model Griffin mixes Hawk layers with local attention (ratio ~6:1 recurrent:attention), hinting that pure attention is overkill and sparse attention + recurrence may be the efficient frontier.

## Related Concepts
[[State Space Model]] [[Mamba]] [[Attention]] [[KV Cache]] [[Transformer]]
