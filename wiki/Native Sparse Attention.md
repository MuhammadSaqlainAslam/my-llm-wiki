---
created: "2026-10-19"
title: "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention"
authors: "Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Y. X. Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, Wangding Zeng (DeepSeek-AI / PKU / UW)"
year: 2025
arxiv: "2502.11089"
citation_count: 561
tags: [attention, sparse-attention, long-context, kernels, hardware, training, deepseek, inference, efficiency]
tldr: "Sparse attention you can actually train and actually speed up: three parallel branches (compressed blocks, top-n selected blocks, sliding window) gated per query, with Triton kernels that load KV by GQA group to keep arithmetic intensity balanced. On a 27B/3B-active MoE, NSA matches or beats full attention on general, long-context, and CoT benchmarks while delivering up to 9.0× forward, 6.0× backward, and 11.6× decode speedups at 64k on A100."
aliases: [NSA, Native Sparse Attention, Natively Trainable Sparse Attention]
theme: efficiency
---

# Native Sparse Attention (NSA)

> Jingyang Yuan, Huazuo Gao, Damai Dai, et al. (DeepSeek-AI, PKU, UW), "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention", 2025 (arXiv:2502.11089)

## The Problem / Motivation

At 64k context, **70–80% of decode latency is attention**. Softmax attention is quadratic in compute and forces a full [[KV Cache]] scan every step — so long-context workloads (repo-level code, multi-turn agents, long CoT) die on the attention path.

The obvious fix is sparsity: most query–key scores are tiny, so skip them. But almost every deployed sparse method fails one of two real requirements:

1. **Hardware-aligned speedup.** FLOP reduction is not wall-clock speedup. Random token gathers thrash [[HBM]], break [[Tensor Cores|Tensor Core]] tiling, and leave one of prefill/decode as slow as full attention (phase-restricted sparsity). Some methods also fight [[GQA]]/[[MQA]]: per-query-head selection multiplies KV traffic even when FLOPs look sparse.
2. **Training-aware design.** Bolt-on inference sparsity distorts a pretrained full-attention model (top 20% of scores often cover only ~70% of mass). Discrete selectors (k-means, SimHash) cut the gradient. Token-granular gathers make backward worse than full attention.

NSA is designed **natively sparse from pretraining**, with kernels that keep blockwise access so it can ride the same [[Flash Attention|FlashAttention]] efficiency curve.

## The Idea

Keep a compact, query-dependent KV set made of **three complementary views of the past** — coarse compressed blocks, fine selected blocks, and a local window — then mix their attentions with a learned gate. Sparsity is hierarchical (compress *and* select), not a single top-k over tokens.

## Architecture / Method

```
query q_t
   │
   ├─► Compression branch  → blockwise mean/MLP over KV blocks (l=32, stride d=16)
   ├─► Selection branch    → top-n raw KV blocks (l′=64, n=16) chosen by compressed scores
   └─► Sliding window      → last w=512 tokens (local precision, no shortcuts)
   │
   ▼
 gated sum:  o* = Σ_c g_t^c · Attn(q_t, K̃^c, Ṽ^c)
```

### 1. Token compression (coarse global map)

Keys/values are pooled in overlapping temporal blocks via a small MLP with intra-block position encoding:

$$
\tilde{K}^{\mathrm{cmp}}_t = \{\varphi(k_{id+1:id+l}) \mid 0 \le i \le \lfloor (t-l)/d \rfloor\}
$$

with block length $l=32$, stride $d<l$ (here 16) so neighboring blocks share context and don’t chop mid-token patterns. This gives a cheap, dense, low-resolution map of the whole prefix.

### 2. Blockwise token selection (fine detail where it matters)

Importance of selection blocks is **not a second score network** — it reuses the compression-branch attention scores $p^{\mathrm{cmp}}_t$ (already computed), remapped to selection blocks of size $l'=64$. For [[GQA]]/[[MQA]] the scores are **summed over the query heads that share a KV group** so every head in the group selects the **same** blocks:

$$
{p^{\mathrm{slc}}_t}' = \sum_{h=1}^{H} p^{\mathrm{slc},(h)}_t
$$

Then keep the **top-$n$ blocks** ($n=16$, always including 1 initial block + 2 local blocks). Contiguous blocks — not random tokens — so [[Kernel fusion]] and Tensor Cores stay happy, matching how FlashAttention tiles memory.

### 3. Sliding window (stop local shortcuts)

Local attention learns first and can starve the other branches. A dedicated window of $w=512$ recent tokens gets its own K/V and its own gate, so compression/selection can specialize in long-range patterns. Independent K/V per branch further blocks gradient interference.

### Kernel design (this is half the paper)

Compression and window can reuse FlashAttention-style kernels. Selection gets a custom Triton kernel:

- **Group-centric loads:** for one query position, pull *all* query heads in a GQA group (they share selected blocks) into [[SRAM]].
- **Shared KV fetch:** stream only the indexed KV blocks into SRAM — one load serves the whole group.
- **Grid over queries:** inner loop length ≈ $n$ blocks is nearly constant, so query/output loops go on the grid scheduler.

Balanced arithmetic intensity: no redundant KV transfers, work spread across SMs. Targets GQA/MQA (what production LLMs actually use), not just MHA.

### Training setup (paper)

27B-total / 3B-active DeepSeekMoE + GQA (4 groups, 64 heads), 30 layers, hidden 2560. Pretrain 270B tokens at 8k, then YaRN long-context adaptation to 32k + SFT. NSA loss curve stays smooth and **slightly under** the full-attention baseline throughout.

## Key Results

| Axis | Result (paper) |
|---|---|
| General benchmarks (MMLU, MMLU-PRO, CMMLU, BBH, GSM8K, MATH, DROP, MBPP, HumanEval) | NSA avg **0.456** vs Full Attn **0.443** (better on 7/9) |
| LongBench avg | NSA **0.469** vs Full Attn **0.437** (also beats H2O / InfLLM / Quest / Exact-Top at matched sparsity) |
| CoT after long SFT | Compared to full attention (baselines can’t train) |
| Forward @ 64k | up to **9.0×** vs full attention |
| Backward @ 64k | up to **6.0×** |
| Decode @ 64k | up to **11.6×** (grows with length; decode is memory-bound so fewer KV bytes ≈ linear win) |
| Hardware | 8× A100; Triton kernels; speedup vs optimized full attention |

Sparsity that shows up in **training, prefill, and decode** — not just one phase.

## Comparison to Prior Work

- vs **H2O / eviction** — drops tokens from the cache; NSA keeps full cache but reads less per query, and trains through the selector.
- vs **Quest / InfLLM / Exact-Top** — inference-time selection on a full-attention pretrained model; NSA is sparse from pretrain, so quality can *exceed* full attention instead of recovering it.
- vs **MInference-style prefill-only sparsity** — NSA speeds prefill *and* decode.
- vs **[[Compressed Sparse Attention]] / [[Heavily Compressed Attention]]** ([[DeepSeek_V4]]) — CSA/HCA compress then select compressed entries (shrink stored cache). NSA’s selection branch reads **uncompressed** top blocks for fine detail while compression covers the long tail — complementary, not identical.
- vs **[[MiniMax-Sparse-Attention|MiniMax Sparse Attention]]** — MSA is single-path block Top-k on GQA; NSA is three-branch (cmp + slc + win) with gated fusion and free importance from the compression scores. MSA later contrasts itself with NSA on exactly this point.
- vs **[[Flash Attention|FlashAttention]] family** — FA is *dense* exact attention with IO-aware tiling; NSA is *sparse* attention with the same blockwise memory discipline so sparse ≠ anti-kernel.

## Limitations

- Three branches + gates + per-branch K/V add parameters and kernel complexity versus a single Top-k selector.
- Selection is block-granular: a critical single token inside a non-selected block is lost unless the compression branch covers it.
- Hyperparameters ($l, d, l', n, w$) are baked into the reported recipe; other regimes are not established here.
- Benchmarks are a 27B/3B-active MoE at ≤32k adapted context (kernels profiled to 64k) — not a dense 100B+ production fleet result.
- Custom Triton selection kernels are Hopper/Ampere-oriented; portability story is “modern GPUs with Tensor Cores + decent SRAM,” not free on every accelerator.

## Why It Matters

This wiki’s running theme is *move fewer bytes per token of intelligence*. NSA is the cleanest modern answer for **long-context attention** on that theme: hierarchical sparsity that (1) **trains end-to-end**, (2) **respects GQA KV sharing**, and (3) **keeps FlashAttention-style block access** so FLOPs saved become milliseconds saved. It sits between pure kernel work ([[Flash Attention]], [[FlashAttention-2]], [[FlashAttention-3]]) and architecture-level cache surgery ([[Compressed Sparse Attention]], [[KV Cache Optimization]]), and it is the reference DeepSeek-style sparse attention that later systems ([[MiniMax-Sparse-Attention|MSA]], MoBA, DSA) explicitly compare against.

## Related Concepts

[[Flash Attention]] · [[FlashAttention-2]] · [[FlashAttention-3]] · [[GQA]] · [[Multi-Query Attention]] · [[KV Cache]] · [[KV Cache Optimization]] · [[Compressed Sparse Attention]] · [[Heavily Compressed Attention]] · [[MiniMax-Sparse-Attention|MiniMax Sparse Attention]] · [[Sliding window attention]] · [[Attention sinks]] · [[Kernel fusion]] · [[Tensor Cores]] · [[SRAM]] · [[HBM]] · [[Prefill]] · [[Decode phase]] · [[DeepSeek_V4]] · [[DeepSeek-V2]] · [[GPU]]