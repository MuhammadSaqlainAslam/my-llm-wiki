---
created: "2026-05-12"
title: "Attention Sinks"
aliases: ["StreamingLLM"]
tags: [attention, kv-cache, streamingllm]
tldr: "The first few tokens of every sequence accumulate disproportionate attention weight across all heads — even when their content is meaningless. Evicting them collapses quality; retaining them lets sliding-window models work at arbitrary context length."
---

# Attention Sinks

An odd empirical fact about Transformers: across many heads in many layers, the attention weight on the first 1–4 tokens of every sequence is unusually large — often dominating other tokens regardless of what's at those positions. The interpretation (from the StreamingLLM paper) is that attention is a softmax — it always sums to 1. When a head has "nothing useful to attend to" for the current query, the leftover attention mass has to go *somewhere*, and the early-position keys (which see plenty of training and develop unusually large norms) become the default sink. This means evicting the first tokens — the obvious thing to do in a sliding-window model — destroys quality. Keep the first 4 + the last $W$ and you can stream indefinitely.

## Where it appears

- [[KV Cache Optimization]]

---

*Related: [[Sliding window attention]] · [[Cache eviction]] · [[Eviction policy]]*
