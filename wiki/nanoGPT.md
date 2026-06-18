---
created: "2026-05-22"
title: nanoGPT
tags: [transformer, language model, GPT-2, training, deep learning, neural network, educational]
year: 2022
tldr: The simplest, fastest repository for training and fine-tuning medium-sized GPTs; a clean rewrite of minGPT that reproduces GPT-2 (124M) on OpenWebText in ~4 days on 8×A100s, in ~300 lines each for training loop and model definition. Now deprecated in favour of nanochat (Nov 2025).
wikilinks: ["[[Mamba]]", "[[FlashAttention-2]]", "[[Multi-Query Attention]]", "[[RoPE]]", "[[Sequence parallelism]]"]
---

# nanoGPT

**Repository:** [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
**Author:** Andrej Karpathy
**Year:** 2022
**Status:** Deprecated (Nov 2025) — superseded by [nanochat](https://github.com/karpathy/nanochat)

## Overview

nanoGPT is a minimal, readable codebase for training and fine-tuning GPT-style [[transformer]] language models. The entire implementation lives in two ~300-line files:

- `train.py` — a straightforward training loop with DDP support
- `model.py` — a GPT model definition that can load OpenAI GPT-2 weights

Its design philosophy prioritises *practicality over pedagogy*, making it easy to hack for custom [[Transformer|language model]] experiments.

## Key Features

| Feature | Detail |
|---|---|
| Baseline reproduce | GPT-2 124M on OpenWebText, val loss ~2.85 |
| Hardware target | Single 8×A100 40 GB node, ~4 days |
| Distributed training | PyTorch DDP (`torchrun`) across multiple nodes |
| Compile support | PyTorch 2.0 `torch.compile` for speed |
| Device support | CUDA, CPU, Apple Silicon MPS |
| Tokenizer | OpenAI `tiktoken` (BPE) |

## Architecture

- Standard decoder-only Transformer (identical to GPT-2)
- Configurable: layers, heads, embedding size, context length (`block_size`)
- Optional weight loading from GPT-2 (up to 1.3B parameters)
- [[RoPE]] and [[FlashAttention-2]] not built-in, but easily added

## Quick Start

```bash
pip install torch numpy transformers datasets tiktoken wandb tqdm
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py
python sample.py --out_dir=out-shakespeare-char
```

## Relation to Other Projects

- Predecessor: **minGPT** (more educational)
- Successor: **llm.c** (C/CUDA, faster, no PyTorch) → see [[llm.c]]
- Tokenizer companion: **minbpe** → see [[minbpe]]
- Autograd engine: **micrograd** → see [[micrograd]]

## Relevance to Wiki Themes

nanoGPT is a canonical reference implementation of the [[transformer]] [[Transformer|neural network]] architecture for [[Transformer|language model]] pretraining. It demonstrates [[Scaling Laws|scaling]] considerations (batch size, context length, model size) and serves as the pedagogical baseline against which hardware-accelerated systems like [[FlashAttention-2]] and [[Sequence parallelism]] are often benchmarked.
