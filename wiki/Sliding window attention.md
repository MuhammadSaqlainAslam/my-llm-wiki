---
created: "2026-05-12"
title: "Sliding Window Attention"
tags: [kv-cache, attention, eviction, locality]
tldr: "Keep only the last $W$ tokens in the KV cache; older tokens fall off the window. The simplest [[Eviction policy]] — destroys long-range recall but bounds memory at $W$."
---

# Sliding Window Attention

The simplest possible [[Cache eviction]] strategy: at every decode step, drop any cache entry older than $W$ tokens. Memory becomes $O(W)$ regardless of total context length. Attention quality degrades smoothly as soon as the relevant context is past the window — so the window has to be sized to fit the longest dependency you care about. Used in Mistral-7B (W = 4096) and a number of other models as a pragmatic compromise. Crucially **does not survive on its own** at long context — you also need to retain the [[Attention sinks]] (the first few tokens), as the StreamingLLM paper showed.

## Where it appears

- [[KV Cache Optimization]]
- [[Step 3.5 Flash]] — interleaves it 3:1 with full attention layers to keep agentic inference fast without losing long-range information entirely

---

*Related: [[Cache eviction]] · [[Eviction policy]] · [[Attention sinks]]*
