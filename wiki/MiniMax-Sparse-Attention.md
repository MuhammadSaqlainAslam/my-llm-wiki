---
created: "2026-10-13"
title: "MiniMax Sparse Attention"
authors: "Xunhao Lai, Weiqi Xu, Yufeng Yang, Qiaorui Chen, Yang Xu, Lunbin Zeng, Xiaolong Li, Haohai Sun, Haichao Zhu, Vito Zhang, Jinkai Hu, Jiayao Li, et al. (MiniMax)"
year: "2026"
arxiv: "2606.13392"
tags: [attention, sparse-attention, gqa, long-context, inference, kernels, gpu]
citation_count: 0
tldr: "Blockwise sparse attention built on GQA: a lightweight Index Branch scores key-value blocks and picks a Top-k subset independently for each GQA group, then a Main Branch runs exact attention over only those blocks. On a 109B natively multimodal MoE it matches GQA quality while cutting per-token attention compute 28.4x at 1M context, with a co-designed kernel giving 14.2x prefill and 7.6x decode speedups on H800. Powers the released MiniMax-M3."
aliases: ["MSA", "MiniMax-M3 attention"]
---

# MiniMax Sparse Attention (MSA)

> Xunhao Lai, Weiqi Xu, Yufeng Yang, Qiaorui Chen, Yang Xu, Lunbin Zeng, Xiaolong Li, Haohai Sun, Haichao Zhu, Vito Zhang, Jinkai Hu, Jiayao Li, et al. (17 authors, MiniMax), "MiniMax Sparse Attention", June 2026 (arXiv:2606.13392)

## The Problem / Motivation

Agentic workflows, repository-scale code reasoning and persistent memory all want a model that attends over hundreds of thousands to millions of tokens at once. Softmax attention is quadratic, so at deployment scale that is untenable: [[Prefill]] cost grows with the square of the prompt, and every [[Decode phase]] step has to read a huge [[KV Cache]]. The question is how to get sparsity that actually turns into wall-clock speedups on real GPUs, not just a smaller FLOP count on paper.

## The Idea

Keep [[GQA]] as the backbone, but stop attending to everything: for each query, retrieve only the handful of key-value *blocks* that matter, and run exact attention over just those. The design principle the authors stress is **simplicity and scalability** — a deliberately streamlined scheme that is straightforward to deploy efficiently across a broad range of GPUs.

## Architecture / Method

```
query ──► Index Branch (lightweight)
            scores key-value blocks, per GQA group
            Top-k blocks selected independently for each group
                       │
                       ▼
          Main Branch: exact block-sparse attention
            over ONLY the selected blocks
```

- **Index Branch.** Scores visible key tokens, aggregates the scores to the block level (max-pooling), and selects a Top-k set of blocks. Selection is made **independently per GQA group** (each key-value head defines one group), so different groups can retrieve different blocks while execution stays block-level and efficient.
- **Main Branch.** Ordinary exact attention, but restricted to the selected blocks.
- **Deployed configuration.** As reported in the paper: block size 128 and k = 16, i.e. about 2,048 tokens attended per query.
- **Kernel co-design.** The paper's second half is as important as the first. Top-k selection is **exp-free** (raw indexer scores go straight to selection, skipping the max/exp/sum steps of softmax), and the sparse attention kernel iterates **KV-outer** with query concatenation so that [[Tensor Cores]] stay busy under block-granular memory access. The authors report their specialized Top-k running about 5.1x faster than `torch.topk` at the deployed setting.
- **Training mechanisms.** An auxiliary KL alignment loss trains the indexer, the Index Branch input is gradient-detached, local blocks are forcibly included, and there is an indexer warm-up phase.

**Two ways to get there.** The paper trains the same 109B-parameter [[Mixture-of-Experts|MoE]] (native text/image/video multimodal training) two ways: **MSA-PT**, sparse from early in pretraining after a 40B-token indexer warm-up, and **MSA-CPT**, continued pretraining from a 2.6T-token GQA checkpoint for 400B more tokens (40B of warm-up, then sparse). So the method can be adopted either from scratch or retrofitted onto an existing GQA model.

## Key Results

| Aspect | Result |
|---|---|
| Quality | Performs on par with GQA on the 109B model (per the abstract) |
| Attention compute | 28.4x lower per-token attention compute at 1M context |
| Prefill | 14.2x wall-clock speedup on H800 |
| Decoding | 7.6x wall-clock speedup on H800 |
| Production use | A natively multimodal production model powered by MSA, MiniMax-M3, is publicly released; inference kernel open-sourced |

Note the gap between 28.4x fewer FLOPs and 7.6–14.2x faster wall-clock: sparsity is not free, and index construction, Top-k selection and gathering overheads eat into the theoretical gain. That gap is exactly why the paper co-designs the kernel.

## Comparison to Prior Work

- vs. **DeepSeek Sparse Attention (DSA)** — DSA scores tokens individually with a multi-head ReLU "lightning indexer" and shares one Top-k index across all query heads; MSA scores whole KV blocks and selects per GQA group.
- vs. **NSA** — NSA targets MQA/MHA backbones with three parallel branches (compressed, selected, sliding-window); MSA keeps one selection path on a GQA backbone.
- vs. **MoBA** — also block-based and GQA-friendly, but scores blocks by averaged keys and trains the indexer only through the language-modeling gradient, with no auxiliary KL loss.
- vs. **this wiki's [[Compressed Sparse Attention]] / [[Heavily Compressed Attention]]** ([[DeepSeek_V4]]) — a related but different design. CSA/HCA *compress* groups of tokens into single KV entries and then select among those, which shrinks the stored cache; MSA *selects among uncompressed blocks*, so it cuts the compute and memory *reads* per token. (Editorial note, not a claim from the paper: since MSA retrieves from the full cache, the cache itself is still stored in full, so its headline win is attention compute and bandwidth rather than cache footprint. The paper does not compare against CSA/HCA.)

## Limitations

- Realized speedups (7.6x decode, 14.2x prefill) are well below the theoretical 28.4x compute reduction because of indexing and gather overheads.
- Quality is reported as on par with GQA overall, but sparse retrieval can lose to full attention on tasks that need information from blocks the indexer fails to select, and the continued-pretraining route is a retrofit of an existing GQA model, so per-benchmark results are worth checking in the paper's tables.
- The kernel is tuned to a particular hardware path (H800 results); gains on other accelerators are not established by the paper.

## Why It Matters

Sparse attention is the other big route (alongside [[Novel attention mechanisms|compressed/shared-KV designs]] and linear/hybrid models) to million-token context. MSA is notable less for a new idea than for being a *shipped* design: it keeps GQA, uses a simple block-and-Top-k recipe, and is validated at 109B MoE scale with a released product model. It slots directly into the wiki's inference-optimization thread — the [[Roofline-Model]] logic of "decode is memory-bound, so read fewer bytes" and the [[KV Cache Optimization]] taxonomy — as a concrete data point for **sparse retrieval over the KV cache**.

## Related Concepts

[[GQA]] · [[KV Cache]] · [[KV Cache Optimization]] · [[Novel attention mechanisms]] · [[Compressed Sparse Attention]] · [[Heavily Compressed Attention]] · [[DeepSeek_V4]] · [[Prefill]] · [[Decode phase]] · [[Tensor Cores]] · [[Roofline-Model]] · [[Mixture-of-Experts]] · [[GPU]]
