---
created: "2026-06-11"
title: "SnapKV: LLM Knows What You Are Looking for Before Generation"
authors: "Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, Deming Chen"
year: "2024"
arxiv: "2404.14469"
aliases: ["SnapKV"]
tags: [kv-cache, compression, inference, long-context]
tldr: "Pools attention weights across all heads on an observation window to vote on which KV positions are globally important, then retains only those — a compact 'snapshot' per layer. Maintains constant decoding speed as context grows and mitigates the long-context slowdown seen in Medusa. NeurIPS 2024."
citation_count: 792
---

## TL;DR

SnapKV observes that, for a given prompt, the attention pattern over past tokens stabilizes before generation even begins — an "observation window" near the end of the prompt reveals which past KV positions the model will keep attending to. SnapKV pools attention weights across all heads within this window to vote on globally important KV positions, then retains only those, discarding the rest. This produces a fixed-size KV "snapshot" per layer, so decoding speed stays constant regardless of how long the input context is. NeurIPS 2024.

## The Problem

As input context grows, the KV cache grows linearly with it, and attention over the full cache at every decoding step gets progressively more expensive — both in memory and in the compute needed for the target model's forward pass. Recency-biased eviction strategies (keep only the most recent tokens) are cheap but throw away semantically important early tokens (e.g. instructions given at the start of a long prompt), degrading quality on tasks that genuinely need that information.

## The Idea

SnapKV's key empirical observation is that attention patterns are largely predictable before generation starts: looking at an "observation window" of the last few tokens of the prompt reveals, via their attention distribution over the rest of the context, which KV positions matter for the upcoming generation. SnapKV pools (aggregates) attention weights across all attention heads over this window to vote on importance, clusters the selected positions to avoid fragmenting contiguous important spans, and retains a fixed-size snapshot of the highest-voted positions per layer — dropping the rest of the KV cache before generation begins.

Because the retained KV cache size is fixed regardless of input length, decoding speed stays constant as context grows, unlike full-cache attention which slows down linearly.

## Key Results

- ~3.6x decoding speedup at long sequence lengths compared to full KV cache attention
- Constant decoding speed independent of input context length, since the compressed cache size doesn't grow with the prompt
- Mitigates a slowdown observed in [[Medusa]] at long sequence lengths — reported ~1.3x speedup over Medusa at 10K-token sequences when combined with it
- Retains semantically important early-context tokens that pure recency-based eviction would discard

## Why It Matters

- The "attention pattern is predictable from an observation window" insight is a reusable primitive that other KV cache compression methods build on
- Directly complementary to speculative decoding methods like [[Medusa]] — addresses their long-context slowdown rather than competing with them
- Retains semantic importance (not just recency), addressing a real quality gap in simpler eviction strategies like [[H2O eviction]]

## Limitations

- The observation-window heuristic assumes the attention pattern near the end of the prompt is representative of what will matter during generation — this can fail for tasks where relevant information is queried unpredictably during generation itself (rather than being knowable from the prompt alone)
- Fixed snapshot size is a hard cutoff; very long generations that shift topic partway through may need context the snapshot already discarded

## Related Concepts

[[KV Cache]] · [[H2O eviction]] · [[PyramidKV]] · [[GQA]] · [[Medusa]] · [[KVQuant]]
