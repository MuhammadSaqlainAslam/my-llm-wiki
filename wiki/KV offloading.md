---
created: "2026-05-12"
title: "KV Offloading"
tags: [kv-cache, paging, dram, nvme, long-context]
tldr: "Move cold KV-cache pages out of [[HBM]] to host DRAM (or NVMe SSD), page them back in when the model needs them. Required when the cache is fundamentally larger than HBM — e.g., 1M-token RAG."
---

# KV Offloading

When a single request's KV cache exceeds the HBM available — common for million-token RAG, long-document QA, or persistent chat sessions — you have to move cache bytes out of HBM. **KV offloading** pages cold blocks out to host DRAM (PCIe latency, ~10 GB/s) or NVMe SSD (~3 GB/s). The page-migration cost is paid only when the model actually needs to attend to a cold block, so locality of attention determines whether offloading is fast or slow. Used in conjunction with [[Paged attention]] (so blocks are uniformly sized and addressable) and often [[Cache compression]] (so transferred bytes are smaller).

## Where it appears

- [[KV Cache Optimization]]

---

*Related: [[Paged attention]] · [[Hybrid memory]] · [[Memory hierarchy]]*
