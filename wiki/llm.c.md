---
created: "2026-05-22"
title: llm.c
tags: [LLM inference, training, CUDA, C, hardware acceleration, GPT-2, deep learning, neural network, flash attention]
year: 2024
tldr: LLM pretraining in pure C and CUDA with no PyTorch or Python dependency; reproduces GPT-2/GPT-3 series and runs ~7% faster than PyTorch Nightly, with a simple ~1000-line CPU fp32 reference alongside a high-performance CUDA path with cuDNN Flash Attention.
wikilinks: ["[[nanoGPT]]", "[[FlashAttention-2]]", "[[Hardware-Aware Scan]]", "[[HBM]]", "[[Systolic array]]", "[[Inference optimization]]", "[[Sequence parallelism]]"]
---

# llm.c

**Repository:** [karpathy/llm.c](https://github.com/karpathy/llm.c)
**Author:** Andrej Karpathy
**Year:** 2024

## Overview

llm.c implements LLM pretraining in **plain C and CUDA**, eliminating the 245 MB PyTorch and 107 MB CPython dependencies. The primary focus is reproducing the GPT-2 and GPT-3 model series. It currently runs approximately **7% faster than PyTorch Nightly** on the same hardware.

The project is a spiritual successor to [[nanoGPT]], sharing its `train_gpt2.py` PyTorch reference implementation as a baseline.

## Code Structure

| File | Description |
|---|---|
| `train_gpt2.cu` | Main CUDA training file (bleeding-edge, mixed precision) |
| `train_gpt2.c` | Simple CPU fp32 reference, ~1,000 lines, single file |
| `train_gpt2.py` | PyTorch reference (tweaked nanoGPT) |
| `test_gpt2.cu` | GPU unit tests vs. PyTorch reference |

## Performance Features

- **[[FlashAttention-2]]** via cuDNN (optional, `USE_CUDNN=1`) — reduces memory and speeds up attention
- **Mixed precision** (fp32 / bf16) training paths
- **Multi-GPU / Multi-node** support for large-scale training
- ~7% wall-clock speedup vs. PyTorch Nightly for GPT-2 124M

## Hardware & Memory

The design is explicitly hardware-aware:
- Targets [[HBM]] bandwidth as the primary bottleneck
- Fused CUDA kernels reduce round-trips through the [[memory hierarchy]]
- cuDNN Flash Attention leverages [[Systolic array]] units in Tensor Cores

## Quick Start (GPU, mixed precision)

```bash
chmod u+x ./dev/download_starter_pack.sh
./dev/download_starter_pack.sh
make train_gpt2cu USE_CUDNN=1
./train_gpt2cu
```

## Quick Start (CPU only)

```bash
make train_gpt2
OMP_NUM_THREADS=8 ./train_gpt2
```

## Datasets

Data scripts in `/dev/data/` download, tokenize (GPT-2 BPE via [[minbpe]]-compatible format), and save tokens as `.bin` files readable from C (uint16 token IDs, 1024-byte header).

## Relation to Wiki Themes

llm.c represents the [[Hardware Acceleration for Neural Networks|hardware acceleration]] and [[Speculative Decoding|LLM inference]]/training frontier: eliminating framework overhead, maximising [[HBM]] bandwidth, and leveraging [[FlashAttention-2]] within a minimal codebase. It is the practical counterpart to theoretical work on [[Inference optimization]], [[Sequence parallelism]], and [[Scaling Laws|scaling]]. The CPU fp32 path serves as an accessible reference analogous to [[micrograd]] for understanding raw forward/backward pass mechanics.
