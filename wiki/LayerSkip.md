---
created: "2026-10-19"
title: "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding"
authors: "Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai, Anas Mahmoud, Bilge Acun, Saurabh Agarwal, Ahmed Roman, Ahmed A Aly, Beidi Chen, Carole-Jean Wu (Meta / CMU)"
year: 2024
arxiv: "2404.16710"
citation_count: 0
tags: [speculative-decoding, inference, training, draft-model, efficiency, self-speculative, early-exit]
tldr: "Train once so early layers exit cleanly (progressive layer dropout + shared early-exit loss), then self-speculate: draft with early layers, verify/correct with the rest of the *same* model. No second draft net — up to 2.16× summarization, 1.82× coding, 2.0× semantic parsing speedups (ACL 2024)."
aliases: [LayerSkip, self-speculative decoding, early exit LLM]
theme: inference
---

# LayerSkip

> Mostafa Elhoushi, et al. (Meta, CMU), "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding", ACL 2024 (arXiv:2404.16710)

## The Problem / Motivation

[[Speculative Decoding]] is lossless speedup: a **separate** [[Draft model|draft model]] proposes tokens; the target verifies. Cost of that recipe:

- Extra model memory and loading
- Draft/verify **don’t share** activations or FLOPs (except at verify)
- Early-exit (stop at layer $k$) on a normally trained LLM is usually bad — later layers carry the real mapping

Want: draft and target **inside one checkpoint**, with early layers actually useful at exit.

## The Idea

**Change the training recipe** so intermediate exits are viable, then use the shallow prefix as the draft and the full stack as the verifier — **self-speculative decoding**.

1. **Progressive layer dropout** — low dropout early, high dropout late (later layers must not rely on a single deep path).
2. **Early exit loss** — all layers share the same exit head; shallow predictions are supervised.
3. **Inference** — generate candidates from an early layer; verify/correct with remaining layers (same KV work, shared compute).

## Architecture / Method

### Training
- Layer dropout rate **increases with depth** (regularizes deep layers harder).
- **One LM head** trained to work at every depth (early exit).
- Works for pretrain-from-scratch, continual pretrain, domain finetune, task finetune (tested on Llama family).

### Self-speculative decode
```
draft  : forward layers 1…k  → propose token(s)
verify : forward layers k+1…L (or full stack) → accept / correct
```
- **Less memory** than a second draft model.
- Draft and verify **share** activations / cache structure (unlike EAGLE/Medusa dual-module setups).
- Lossless in the speculative-decoding sense (verify step decides).

### Also: early-exit inference
For easy tokens / low-stakes steps, stop at a shallow exit when confidence is high (throughput mode).

## Key Results

| Task | Speedup |
|---|---|
| CNN/DM summarization | up to **2.16×** |
| Coding | up to **1.82×** |
| TOPv2 semantic parsing | up to **2.0×** |
| Memory | **smaller** than draft-model speculative methods |

Open-source code + checkpoints (ACL 2024).

## Comparison to Prior Work

- vs **[[Speculative-Decoding-Leviathan\|Speculative decoding]] (separate draft)** — LayerSkip drafts *inside* the target; no second model.
- vs **[[Medusa]]** — Medusa adds extra heads on a frozen LLM; LayerSkip reshapes **depth** training (exit loss + layer dropout).
- vs **[[EAGLE]] / [[EAGLE-2]] / [[EAGLE-3]]** — feature-level draft module; LayerSkip is self-speculative with shared stack (often less memory).
- vs **[[QuantSpec]]** — self-speculative via quantized weights/KV; LayerSkip via **early layers**. Complementary axes.
- vs naive **early exit** without the recipe — those exits are weak; LayerSkip’s dropout + shared exit loss is what makes them draftable.

## Limitations

- Needs **training or finetune** (or continual pretrain) — not a drop-in kernel for an existing HF checkpoint.
- Draft quality from layer $k$ is task-dependent (summarization/coding gains ≠ every domain).
- Early-exit “just stop” mode is approximate; self-spec path is the lossless story.
- Tuning $k$ (exit depth) and dropout schedule is empirical.
- Gains are wall-clock on their Llama stack; other kernels/architectures may shift the ×.

## Why It Matters

Closes a real gap in this wiki’s inference cluster: **draft without a second model**. LayerSkip is the clean “self-speculative / early exit” reference next to [[Medusa]] and [[EAGLE]], and it shows training-time design can unlock inference-time parallelism — same spirit as [[BitNet b1.58|BitNet]] changing the forward math and [[Native Sparse Attention|NSA]] making sparsity trainable.

## Related Concepts

[[Speculative Decoding]] · [[Speculative-Decoding-Leviathan]] · [[Speculative-Sampling-Chen]] · [[Draft model]] · [[Verification step]] · [[Medusa]] · [[Medusa heads]] · [[EAGLE]] · [[EAGLE-2]] · [[EAGLE-3]] · [[QuantSpec]] · [[HYDRA]] · [[Lookahead Decoding]] · [[Lossless speedup]] · [[LayerSkip]] · [[Prefill]] · [[Decode phase]] · [[Transformer]]