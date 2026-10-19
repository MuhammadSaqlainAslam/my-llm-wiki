---
created: "2026-10-19"
title: "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision"
authors: "Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao"
year: 2024
arxiv: "2407.08608"
citation_count: 752
tags: [attention, hardware, kernel-fusion, gpu, hopper, fp8, asynchrony, warp-specialization, efficiency]
tldr: "FlashAttention for Hopper: warp-specialized producer/consumer pipelines + pingpong GEMM↔softmax overlap + FP8 with block quantization and incoherent processing. Same exact-attention math as FA2, but 1.5–2.0× faster FP16 (up to 740 TFLOPS, ~75% H100 util) and ~1.2 PFLOPS in FP8 — with 2.6× lower numerical error than naive FP8 attention."
aliases: [FlashAttention-3, FlashAttention v3, FA3]
theme: hardware
---

# FlashAttention-3

> Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao, "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision", 2024 (arXiv:2407.08608)

## TL;DR

[[Flash Attention|FlashAttention v1]] fixed **bytes** (tile attention into [[SRAM]], never write $T \times T$ to [[HBM]]). [[FlashAttention-2]] fixed **work distribution** (50–73% of A100 peak). But on H100, FA2 still only hits **~35% utilization** while tuned GEMMs reach 80–90%. The gap is no longer “bad tiling” — it’s that FA2 ignores the two features Hopper actually added: **asynchrony** (TMA loads and Tensor Core WGMMAs run in the background) and **low precision** (FP8 Tensor Cores ≈ 2× FP16 throughput).

FlashAttention-3 keeps the same exact online-softmax algorithm and rewrites the schedule:

1. **Warp-specialized producer/consumer** + circular SMEM buffer (TMA vs WGMMA)
2. **Pingpong scheduling** — one warpgroup’s softmax under another’s GEMMs
3. **Intra-warpgroup 2-stage GEMM↔softmax pipeline**
4. **FP8 GEMMs** with block quantization + incoherent processing to tame outliers

**Results on H100 SXM5:** FP16 **1.5–2.0×** over FA2 forward (up to **740 TFLOPS**, ~**75%** util), **1.5–1.75×** backward; FP8 near **1.2 PFLOPS**. FP16 matches FA2 numerical error; FP8 FA3 is **2.6×** more accurate than naive per-tensor FP8 attention on outlier-heavy cases.

---

## The Core Idea — What Hopper Changed

FA2 assumed a mostly synchronous kernel: load tile → matmul → softmax → matmul → store. Hopper splits those roles in hardware:

| Unit | Job | Async? |
|---|---|---|
| **TMA** | bulk GMEM ↔ SMEM copies | yes |
| **Tensor Core (WGMMA)** | matmul, can read operands from SMEM | yes |
| **CUDA cores / MUFU** | exp, multiplies, rescaling | throughput-starved |

Softmax is not “slow math” in FLOPs — it’s slow in *special-function throughput*. On H100 FP16 (head dim 128) the forward pass has ~512× more matmul FLOPs than exp FLOPs, but exp has ~256× lower throughput, so **exp can eat ~50% of the cycles** next to matmul if serialized. FP8 doubles matmul rate and leaves exp alone — the imbalance gets worse.

FA3’s job: keep Tensor Cores and TMA busy while softmax hides in the gaps.

---

## Three Techniques

### 1. Producer–consumer warp specialization

Split each CTA’s warps:

- **Producer warpgroup** — only issues TMA loads of $Q_i$, then $\{K_j, V_j\}$ into an $s$-stage circular SMEM buffer (`setmaxnreg` frees registers).
- **Consumer warpgroup** — only issues WGMMAs + softmax on tiles already in SMEM (reallocates those registers).

No warp both loads and computes. The compiler gets a clean schedule; TMA latency hides under WGMMA and vice versa. Same tiling math as FA2 (blocks $B_r \times d$ of $Q$, $B_c \times d$ of $K,V$, online softmax with running max $m_i$ and sum $\ell_i$).

### 2. Pingpong scheduling (across warpgroups)

Barriers (`bar.sync`) force **GEMMs of warpgroup 1 → then GEMMs of warpgroup 2**, so *softmax of WG1* runs while *WG2’s WGMMAs* are in flight — then swap. Softmax of one group is scheduled on the multi-function unit while the other group saturates Tensor Cores.

Even before intra-warpgroup overlap, this alone moved FP16 forward (head dim 128, seq 8192) from **~570 → 620–640 TFLOPS**.

MQA/GQA: keep FA2’s indexing so $K,V$ are not duplicated in HBM.

### 3. Intra-warpgroup 2-stage GEMM↔softmax pipeline

Within one consumer iteration, $S = QK^\top$ must finish before softmax, and $\tilde{P}V$ needs $\tilde{P}$. FA2 waits on those dependencies. FA2’s wait is what FA3 breaks **across iterations**:

- Issue WGMMA for $S_{\mathrm{next}}$ (commit, don’t wait)
- Softmax / exp / rescale on $S_{\mathrm{cur}}$ while that WGMMA runs
- Issue $\tilde{P}_{\mathrm{cur}} V_{j-1}$ overlapping with softmax of the next block

A 2-stage pipeline keeps an extra $S_{\mathrm{next}}$ in registers ($B_r \times B_c$ floats per CTA) — a real cost against large tiles. A 3-stage variant was tried and **lost**: NVCC reorders so only the first WGMMA overlaps, and register pressure forces smaller tiles. 2-stage is the sweet spot the paper ships (SASS dump confirms exp2 / rowsum / O-rescale interleave with HGMMA).

### 4. FP8 without wrecking accuracy

Hopper FP8 WGMMA nearly doubles GEMM TFLOPS but:

- Operands must be **k-major** (contiguous in the contraction dim) — clashes with typical attention layouts and with FP32 accumulator layouts when chaining GEMMs.
- LLM activations have **outlier features**; naive per-tensor FP8 quant is noisy.

FA3’s answers:

- **Layout transforms** so chained FP8 WGMMAs can fire (V tiles oriented for the second GEMM, etc.).
- **Block quantization** — scale per block, not one global tensor scale.
- **Incoherent processing** — spread outlier energy (Hadamard-like) before quant so no single channel dominates the scale.

Measured: FA3-FP8 error is **2.6× lower** than standard FP8 attention with per-tensor quantization when outliers are present. FP16 FA3 matches FA2 error (softmax still in FP32).

---

## Key Results (H100 SXM5 80GB)

| Setting | Result |
|---|---|
| FA2 utilization on H100 | ~**35%** (motivating number) |
| FA3 FP16 forward | **1.5–2.0×** over FA2, up to **740 TFLOPS** (~**75%** util) |
| FA3 FP16 backward | **1.5–1.75×** over FA2 |
| FA3 FP8 | ≈ **1.2 PFLOPS** peak class |
| vs cuDNN attention (long seq) | FP16 FA3 **faster**; FP8 competitive (head dim 64 often ahead; 128/256 depends on causal mask) |
| Numerics | FP16 = FA2 error; FP8 **2.6×** better than naive FP8 |
| Pingpong alone (hdim 128, seq 8k) | ~570 → **620–640 TFLOPS** FP16 forward |

Open-sourced in `Dao-AILab/flash-attention` (permissive license).

## Comparison to Prior Work

- vs **[[Flash Attention]] (v1)** — v1: IO-aware exact attention, no $T \times T$ in HBM. FA3 keeps that contract.
- vs **[[FlashAttention-2]]** — v2: better parallelism on Ampere (25–40% → 50–73% A100). FA3: same algorithm class, Hopper asynchrony + FP8 (35% → ~75% H100).
- vs **cuDNN / vendor attention** — FA3 is competitive or better on long sequences in FP16 and open source.
- vs **approximate sparse attention** ([[Native Sparse Attention|NSA]], [[MiniMax-Sparse-Attention|MSA]]) — FA3 is still **exact** dense attention; those methods change *which* keys are read. FA3 is the dense floor every sparse method must beat on wall-clock for the dense part of the workload (or reuse as their inner kernel, as NSA does for compressed/window branches).
- vs **training quantization** ([[FP8-Training]], [[Quantization]]) — FA3 quantizes **attention GEMMs at kernel level** with block + incoherent tricks; it is not a full-model recipe like [[FP8-Training]].

## Limitations

- Tuned for NVIDIA Hopper (H100) class features: TMA, WGMMA, `setmaxnreg`, FP8 Tensor Cores. Ampere gains are smaller; other vendors need a rewrite of the async story.
- 2-stage pipeline register cost fights large $B_r \times B_c$ tiles — tile vs overlap is a profile-driven tradeoff.
- 3-stage overlap exists in the paper but **underperforms** 2-stage (compiler + registers).
- FP8 forward is the headline; users who cannot accept any quantization error stay on FP16 path (still much faster than FA2).
- Still exact dense attention — does not by itself solve $O(n^2)$ *work* or KV *capacity*; pair with [[GQA]], [[KV Cache Optimization]], or [[Native Sparse Attention|NSA]] for those.

## Why It Matters

If FlashAttention’s thesis is *optimize bytes and let the math stay exact*, FlashAttention-3 is the sequel thesis: **optimize the schedule for the actual async hardware, and use low precision only where you can prove the error budget**. It completes the FA lineage in this wiki ([[Flash Attention]] → [[FlashAttention-2]] → FA3) and is the default dense attention kernel every modern serving stack and every sparse-attention paper (including NSA) either uses or is measured against. The pingpong + producer/consumer ideas also transfer to any kernel where a high-throughput matmul unit must hide a low-throughput elementwise stage — which is most of deep learning.

## Related Concepts

[[Flash Attention]] · [[FlashAttention-2]] · [[Online softmax]] · [[Kernel fusion]] · [[IO-awareness]] · [[Tiling]] · [[Warp]] · [[Occupancy]] · [[Tensor Cores]] · [[SRAM]] · [[HBM]] · [[GPU]] · [[FP8-Training]] · [[Quantization]] · [[GQA]] · [[Multi-Query Attention]] · [[Native Sparse Attention]] · [[Hardware-aware algorithms]] · [[Hardware Acceleration for Neural Networks]]