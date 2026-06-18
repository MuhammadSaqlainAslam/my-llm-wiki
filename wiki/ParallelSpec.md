---
created: "2026-06-11"
title: "ParallelSpec"
aliases: ["ParallelSpec"]
year: 2025
tags: [speculative-decoding, parallelism, inference, stub]
tldr: "Parallel speculative verification: verifies multiple draft sequences simultaneously across different GPU streams rather than sequentially."
---

## TL;DR
Standard speculative decoding verifies one draft sequence at a time. ParallelSpec batches multiple independent draft sequences for simultaneous verification, improving GPU utilization when draft acceptance rate is low.

## See Also
[[Speculative Decoding]] · [[Multi-Token Prediction]] · [[Medusa]]
