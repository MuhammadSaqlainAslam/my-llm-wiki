---
created: "2026-06-11"
title: "LongSpec"
aliases: ["LongSpec"]
year: 2025
tags: [speculative-decoding, long-context, inference, stub]
tldr: "Speculative decoding adapted for long-context inference: the draft model uses a compressed KV representation to stay fast at 128k+ token contexts."
---

## TL;DR
At 128k+ context, the draft model's KV cache dominates memory. LongSpec compresses the draft's KV using learned token merging, enabling speculative decoding at long context without the memory overhead of a full second KV cache.

## See Also
[[Speculative Decoding]] · [[KV Cache]] · [[GQA]] · [[Multi-Token Prediction]]
