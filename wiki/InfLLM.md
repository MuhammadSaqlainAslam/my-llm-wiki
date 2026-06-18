---
created: "2026-06-11"
title: "InfLLM"
aliases: ["InfLLM"]
year: 2024
tags: [kv-cache, offloading, long-context, inference, stub]
tldr: "Infinite LLM inference via KV offloading: streams the KV cache to CPU/NVMe, retrieving only the attention-relevant blocks per step via a block-level index."
---

## TL;DR
InfLLM extends context to 1M+ tokens by offloading the KV cache to CPU memory (or NVMe) and maintaining a block-level semantic index. For each decoding step it retrieves only the top-K relevant blocks — keeping GPU memory use O(K) regardless of total context length.

## See Also
[[KV Cache]] · [[Attention Sinks]] · [[MagicPIG]] · [[GQA]]
