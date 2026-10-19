---
created: "2026-10-19"
title: "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (BitNet b1.58)"
authors: "Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, Furu Wei (Microsoft Research / UCAS)"
year: 2024
arxiv: "2402.17764"
citation_count: 35
tags: [quantization, efficiency, 1-bit, training, inference, energy, hardware, architecture]
tldr: "Train a Transformer whose every weight is ternary {−1, 0, 1} (1.58 bits) and 8-bit activations — matching FP16 quality from 3B up while cutting latency, memory, and energy hard (3B: 2.7× faster / 3.6× less memory; 70B: 4.1× faster / ~8.9× throughput; ~71× matmul energy on 7nm). Not post-training quant: a from-scratch recipe that wants new hardware."
aliases: [BitNet b1.58, BitNet, 1-bit LLMs, Era of 1-bit LLMs]
theme: quantization
---

# BitNet b1.58 (The Era of 1-bit LLMs)

> Shuming Ma, Hongyu Wang, et al. (Microsoft Research), "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits", 2024 (arXiv:2402.17764)

## TL;DR

LLMs are mostly `nn.Linear`. FP16 matmuls are multiply-adds. If every weight is in $\{-1, 0, +1\}$, matmul becomes **integer add / subtract / skip** — no multiplies. BitNet b1.58 does that **from pretraining** (not post-training quant), keeps LLaMA-style blocks ([[RMSNorm]], SwiGLU, [[RoPE]], no biases), and **matches FP16 LLaMA quality starting at 3B** while being dramatically cheaper to serve.

The “1.58 bits” is $\log_2 3 \approx 1.58$ for three symbols. The extra $0$ vs original 1-bit BitNet is feature filtering (explicit “don’t use this input”) and closes the quality gap.

## The Problem / Motivation

Two separate costs dominate inference:

1. **Compute energy** — FP16 mul+add on 7nm is expensive; power is the real FLOPS ceiling ([[Hardware Acceleration for Neural Networks]]).
2. **Memory traffic** — weights and [[KV Cache]] move DRAM→SRAM every step; decode is bandwidth-bound.

Post-training quantization (GPTQ/AWQ/[[Quantization]] family) helps but is suboptimal: you train big then shrink. BitNet argues you should **train in the low-bit regime** so the network can use zeros and ternary geometry natively.

## The Idea

Replace every `nn.Linear` with `BitLinear`:

- **Weights:** absmean scale then round-clip to $\{-1,0,1\}$:

$$
\tilde{W} = \mathrm{RoundClip}\!\left(\frac{W}{\gamma+\epsilon}, -1, 1\right),\quad
\gamma = \frac{1}{nm}\sum_{ij}|W_{ij}|
$$

- **Activations:** 8-bit, scaled per token to $[-Q_b, Q_b]$ (no zero-point).
- **Everything else LLaMA-alike** so stacks like HuggingFace / vLLM / llama.cpp can adopt it with minimal glue.

Forward: scale → ternary matmul (adds) → 8-bit activations. That is the whole computation paradigm — and why the paper calls for 1-bit-native chips (beyond [[LPU]]-style designs).

## Key Results

| Model | PPL (WikiText2-ish) | vs FP16 LLaMA |
|---|---|---|
| 700M | 12.87 vs 12.33 | small gap; **2.6×** less mem, **1.23×** faster |
| 1.3B | 11.29 vs 11.25 | near-parity; **2.9×** mem, **1.67×** speed |
| **3B** | **9.91 vs 10.04** | **matches/beats**; **3.55×** mem, **2.71×** speed |
| 3.9B | 9.62 | **beats 3B FP16** with less mem/latency |

Zero-shot (ARC-e/c, HellaSwag, BoolQ, OBQA, PIQA, WinoGrande): gap **closes with scale**; 3B BitNet ≈ 3B LLaMA avg (50.2 vs 49.7); 3.9B > 3B FP16.

**Serving / energy:**

| Metric | Result |
|---|---|
| 70B decode latency | **4.1×** faster than FP16 LLaMA (2-bit kernel) |
| 70B throughput (2×A100-80GB) | max batch **176 vs 16** (11×); **8.9× tokens/s** (2977 vs 333) |
| Matmul energy (7nm) | **~71.4×** lower than FP16 mul-add mix |
| Implied iso-cost scaling | 13B BitNet < 3B FP16 cost; 30B < 7B; 70B < 13B (latency, mem, energy) |

**Scale-out in tokens:** 3B BitNet @ **2T** tokens (StableLM-3B recipe) **beats** StableLM-3B-3B on Winogrande / PIQA / SciQ / LAMBADA / ARC-e (avg **74.34 vs 73.22**).

Paper also flags: 1-bit **MoE** (tiny weight mem + less all-to-all), longer [[KV Cache]] via 8-bit (then lower) activations, edge/CPU friendliness, and dedicated 1-bit hardware.

## Comparison to Prior Work

- vs **post-training quant** (GPTQ, AWQ, SmoothQuant, [[KVQuant]] for KV) — those compress a trained FP model; b1.58 is **native** ternary training. PTQ remains useful on existing checkpoints; BitNet is a new recipe.
- vs **[[FP8-Training]] / [[NVFP4]]** — still floating multiplies; BitNet removes muls at the matmul core.
- vs **original BitNet (1-bit)** — adding `0` enables feature filtering and reaches FP16 parity from 3B.
- vs **[[QuantSpec]] / quantized drafts** — those quantize for speculative *inference*; BitNet quantizes the *weights of the model itself* from scratch.

## Limitations

- Quality match is claimed **from 3B** — smaller models still trail FP16 a bit.
- Embeddings / some paths stay higher precision; “every parameter ternary” is the Linear story.
- Kernels used (Ladder 2-bit) are not a full 1-bit-native stack; the paper *asks* for new hardware — GPUs today are not optimal.
- Published recipe is 100B-token controlled LLaMA comparison + one 2T 3B run — not a full multi-trillion frontier report.
- Downstream chat / RL / long-context behavior not the focus here.
- MoE + 1-bit is argued, not fully trained at scale in this paper.

## Why It Matters

If [[Flash Attention]] says *move fewer bytes* and [[Mamba]] says *compress context*, BitNet says *make each byte and each multiply cheaper at the source*. It is the landmark “train 1.58-bit, don’t just quantize later” result, and it reframes [[Quantization]] from a deployment afterthought to an **architecture + systems co-design** (with real Pareto wins: better 3.9B than 3B FP16). Any serious efficiency curriculum in this wiki — [[Hardware Acceleration for Neural Networks]], [[NVFP4]], [[KV Cache Optimization]], edge LLMs — needs this note.

## Related Concepts

[[Quantization]] · [[FP8-Training]] · [[NVFP4]] · [[KVQuant]] · [[QuantSpec]] · [[Hardware Acceleration for Neural Networks]] · [[LPU]] · [[SRAM]] · [[HBM]] · [[RMSNorm]] · [[SwiGLU]] · [[RoPE]] · [[Transformer]] · [[Mixture-of-Experts]] · [[KV Cache]] · [[vLLM]]