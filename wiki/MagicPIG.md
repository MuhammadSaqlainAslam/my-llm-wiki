---
created: "2026-06-11"
title: "MagicPIG"
aliases: ["MagicPIG"]
year: 2024
tags: [kv-cache, lsh, inference, sparse-attention, stub]
tldr: "LSH-based KV cache selection: uses locality-sensitive hashing to retrieve the top-K most relevant KV pairs per query without scanning the full cache."
---

## TL;DR
Instead of evicting KV entries, MagicPIG keeps the full cache but retrieves only the top-K most relevant entries per query step using LSH (locality-sensitive hashing). This turns exact attention into approximate retrieval — O(K) cost regardless of sequence length.

## See Also
[[KV Cache]] · [[SnapKV]] · [[Approximate Attention]] · [[Speculative Decoding]]
