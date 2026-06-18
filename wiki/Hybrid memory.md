---
created: "2026-05-12"
title: "Hybrid Memory (KV-Cache Tiering)"
tags: [kv-cache, paging, hbm, dram, nvme]
tldr: "Tier the KV cache across [[HBM]] / host DRAM / SSD, paging hot blocks into HBM and offloading cold blocks. The third family of KV-cache optimization — exemplified by PagedAttention, vAttention, and NVMe offloading."
---

# Hybrid Memory (KV-Cache Tiering)

When the KV cache exceeds HBM capacity, you have two choices: shrink it ([[Cache compression]] or [[Cache eviction]]) or *page* it across slower memory tiers. Hybrid-memory approaches lay the cache out in fixed-size pages (PagedAttention's standard is 16 tokens per page), keep the active pages in HBM, and migrate cold pages to host DRAM or NVMe. The tier hierarchy ([[Memory hierarchy]]) gives you tens of GB → hundreds of GB → terabytes of effective cache capacity at the cost of latency on the colder tiers. Critical for million-token contexts where the cache is fundamentally larger than HBM.

## Where it appears

- [[KV Cache Optimization]]

---

*Related: [[Paged attention]] · [[KV offloading]] · [[Memory hierarchy]]*
