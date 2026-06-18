---
created: "2026-05-12"
title: "Paged Attention"
tags: [kv-cache, paging, vllm, fragmentation]
tldr: "Store the KV cache in fixed-size blocks (typically 16 tokens per page), like an OS virtual-memory page table. Eliminates HBM fragmentation, enables prefix sharing across requests, and is the foundation of vLLM's serving architecture."
---

# Paged Attention

Naive KV-cache allocation gives each request a contiguous HBM region sized for its maximum context length — wasteful if the request is shorter, and prone to fragmentation across many concurrent requests. **PagedAttention** (Kwon et al. 2023, the basis of vLLM) borrows the operating-systems trick: chop the cache into fixed-size blocks (16 tokens × $h_kv$ heads × $d$ dimensions per page), maintain a per-request page table, and allocate pages on demand. Two large wins: (i) zero fragmentation — pages are uniform, allocator is trivial — and (ii) **prefix sharing**, where multiple requests with the same prompt prefix share the same physical pages, dramatically improving throughput for chat and tool-calling workloads. The de-facto standard for production LLM serving.

## Where it appears

- [[KV Cache Optimization]]

---

*Related: [[KV Cache]] · [[Hybrid memory]] · [[Prefill]]*
