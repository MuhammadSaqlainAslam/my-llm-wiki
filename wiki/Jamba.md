---
title: "Jamba: A Hybrid Transformer-Mamba Language Model"
authors: "Opher Lieber, Barak Lenz, Hofit Bata, Gal Cohen, Jhonathan Osin, Itay Dalmedigos, et al."
year: "2024"
arxiv: "2403.19887"
tags: [llm, architecture, mamba, ssm, transformer, mixture-of-experts, hybrid, long-context, inference-efficiency]
tldr: "Jamba combines Transformer attention layers with Mamba SSM layers and MoE in a 1:7 attention-to-Mamba ratio, achieving 52B total / 12B active parameters that fit on a single 80GB GPU with a 4GB KV cache at 256K context, matching Mixtral-8x7B quality at 3x throughput."
citation_count: 441
aliases: ["Jamba", "Jamba-v0.1"]
---

## TL;DR
Jamba is the first production-grade hybrid Attention-SSM model, interleaving Transformer and Mamba layers at a 1:7 ratio with MoE applied every 2 layers (16 experts, top-2). The 52B total / 12B active parameter model fits on a single 80GB GPU even at 128K+ token contexts, achieves 3x throughput over Mixtral-8x7B on long sequences, and matches Mixtral and Llama-2-70B on standard benchmarks while supporting a 256K token context window.

---

## The Problem

Transformers dominate language modeling but have two fundamental scaling problems:

1. **KV cache blows up with context length**: A Mixtral-8x7B or Mistral-7B model requires **32GB of KV cache** at 256K context (16-bit). Llama-2 requires **128GB** — far exceeding any single GPU.
2. **Inference is slow for long contexts**: Every generated token attends over the entire context, so compute scales linearly with sequence length. Throughput degrades badly.

Pure RNNs/SSMs (Mamba, etc.) don't have a KV cache but:
- **Lag behind Transformers in quality** at comparable scale.
- Prior hybrid attempts (H3 at 2.7B, StripedHyena-7B) failed to match attention-only baselines like Mistral-7B.

No prior work had demonstrated a hybrid SSM-Attention model that simultaneously: (a) matched Transformer quality at production scale, (b) handled 256K context, and (c) fit on a single GPU.

---

## Core Innovation

**Intuition**: Attention is the "memory" component — precise but expensive. Mamba is the "processing" component — cheap but imprecise over long range. Use attention sparingly (1 in every 8 layers) to maintain quality, let Mamba handle the bulk of computation, and use MoE to boost capacity without proportionally increasing compute or memory.

The key insight is that **you don't need attention at every layer**. A 1:7 attention-to-Mamba ratio gives you most of the quality benefit of attention (long-range recall, in-context learning) while:
- Reducing KV cache by **8x** (only 1/8 of layers produce KV cache entries)
- Boosting throughput **3x** on long sequences (Mamba layers are O(L) vs O(L²) for attention)
- Keeping the model on a single GPU

MoE then decouples **capacity** (total params = 52B) from **compute** (active params = 12B), analogous to how Mixtral-8x7B works.

---

## Architecture / Method

### The Jamba Block

Jamba stacks 4 identical **Jamba blocks** in sequence. Each block has:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| $l$ | 8 | Layers per block |
| $a:m$ | 1:7 | 1 attention layer per 7 Mamba layers |
| $e$ | 2 | MoE applied every 2 layers |
| $n$ | 16 | Total experts per MoE layer |
| $K$ | 2 | Top-K experts selected per token |

Total: 4 blocks × 8 layers = **32 layers**, with 4 attention layers and 28 Mamba layers.

### Layer Types

Each layer = (Attention or Mamba module) + (MLP or MoE):

```
Types of layers:
1. Mamba layer:           RMSNorm → Mamba → RMSNorm → MLP
2. Mamba MoE layer:       RMSNorm → Mamba → RMSNorm → MoE
3. Transformer layer:     RMSNorm → Attention → RMSNorm → MLP
4. Attention MoE layer:   RMSNorm → Attention → RMSNorm → MoE
```

In the released model, attention MoE layers are not used. MoE is applied every $e=2$ layers, giving ~8 effective experts on average per layer position across the block.

### Mamba Layer Details

Mamba is a selective SSM. The recurrent state at position $t$ is:

$$h_t = A h_{t-1} + B x_t$$
$$y_t = C h_t$$

where $A$, $B$, $C$ are input-dependent (selective). Key property: the entire context is compressed into a **fixed-size hidden state** — no KV cache needed. Jamba adds **RMSNorm inside Mamba layers** to stabilize training at scale (critical finding).

### MoE Module

Standard sparse MoE: router computes softmax over $n=16$ experts, selects top $K=2$:

$$\text{output} = \sum_{i \in \text{top-K}} g_i \cdot E_i(x)$$

Load balancing loss applied (standard). With 16 experts and top-2, only 12.5% of expert parameters are active per token.

### Other Architecture Choices

- **GQA** (Grouped Query Attention) for further KV cache reduction
- **SwiGLU** activation
- **No positional embeddings** (RoPE etc. not needed — Mamba handles position implicitly)
- Vocabulary size: **64K** with BPE; each digit is a separate token
- Tokenizer: no dummy space prefix (cleaner than LLaMA/Mistral tokenizers)

### KV Cache Comparison

At 256K context, 16-bit:

| Model | Total Params | Active Params | KV Cache |
|-------|-------------|---------------|----------|
| LLaMA-2 | 6.7B | 6.7B | **128GB** |
| Mistral | 7.2B | 7.2B | 32GB |
| Mixtral | 46.7B | 12.9B | 32GB |
| **Jamba** | **52B** | **12B** | **4GB** |

The 8x reduction comes directly from having only 1/8 of layers be attention layers.

---

## Key Results

### Standard Benchmarks

| Benchmark | Jamba | Mixtral-8x7B | Llama-2-70B |
|-----------|-------|--------------|-------------|
| HellaSwag | ~83% | ~81% | ~83% |
| ARC-Challenge | ~59% | ~59% | ~57% |
| WinoGrande | ~82% | ~81% | ~80% |
| MMLU | ~67% | ~70% | ~69% |
| GSM8K | ~59% | ~58% | ~54% |
| HumanEval | ~29% | ~27% | ~26% |

*Jamba is broadly competitive with Mixtral-8x7B and Llama-2-70B despite fitting on a single GPU.*

### Long-Context Evaluations

| Context Length | Jamba vs Mixtral |
|---------------|------------------|
| Up to 256K tokens | Jamba outperforms Mixtral on most datasets |
| 128K+ | Jamba fits on single GPU; Mixtral cannot |

### Throughput

- **3x throughput** over Mixtral-8x7B on long contexts
- At 256K context: Jamba produces 4GB KV cache vs Mixtral's 32GB (8x reduction)
- Fits in single 80GB GPU with int8 weights even at 128K+ token contexts

### Ablation: Attention-to-Mamba Ratio

The 1:7 ratio was selected as the **most compute-efficient variant among best-performing variants**. Fewer attention layers hurt quality; more attention layers hurt throughput without quality gains.

---

## Why It Matters

- **Single GPU deployment of a 52B parameter model**: With int8 weights and only 4GB KV cache overhead at long contexts, Jamba is the first model of this scale deployable on a single A100 80GB for long-context inference.
- **256K context window** at production quality: No prior open model of comparable quality supported this; Jamba achieves it while outperforming Mixtral on long-context tasks.
- **3x throughput gain** over Mixtral-8x7B on long sequences, directly from the 1:7 attention-to-Mamba ratio reducing the attention compute bottleneck.
- **Validated the hybrid architecture at scale**: Prior hybrid SSM-attention models (StripedHyena-7B, H3) failed to match attention-only baselines. Jamba is the first to convincingly succeed at production scale.
- **Open source (Apache 2.0)** with planned release of ablation checkpoints, enabling the community to explore this novel architecture family systematically.

---

## Connections to Other Work

### Builds On
- [[Mamba]] — The SSM backbone; Jamba uses Mamba layers with added RMSNorm for stability
- [[Mixtral|Mixtral-8x7B]] — MoE design inspiration; Jamba matches its quality with 8x smaller KV cache
- [[Transformer|Transformer Architecture]] — Attention layers retained at 1:8 ratio
- [[HIPPO|H3]] — Early hybrid SSM+attention up to 2.7B params; Jamba scales this to production
- [[Mixture of Experts]] — Sparse MoE (top-2 of 16) to decouple capacity from compute

### Competes With
- [[Mixtral|Mixtral-8x7B]] — Similar active params (12B vs 12.9B), similar quality, but Jamba has 3x throughput and 8x smaller KV cache
- [[LLaMA 2|Llama-2 70B]] — Comparable quality, but Llama-2-70B requires much more memory and cannot do 256K context
- [[StripedHyena-7B]] — Another hybrid SSM-attention model; lags Mistral-7B, unlike Jamba

### Enables / Related
- [[State Space Model|State Space Models]] — Validates SSM viability in production LLMs
- [[KV Cache Optimization|Long-Context LLMs]] — New benchmark for open models at 256K tokens
- [[Efficient LLM Inference]] — KV cache compression via architectural design rather than post-hoc compression

---

## Limitations

1. **Base model only**: Released without instruction tuning or RLHF alignment. Not suitable for end-user deployment without additional fine-tuning.
2. **Mamba quality ceiling**: Despite the 1:7 ratio working well, the model still slightly underperforms on MMLU (~67% vs Mixtral's ~70%), suggesting pure-attention models may still have quality advantages on knowledge-intensive tasks.
3. **Ablations only at ≤7B parameters**: The architectural design choices (ratio, MoE frequency) were validated at up to 7B params / 250B tokens. Whether the same choices are optimal at 52B+ scale is not fully verified.
4. **MoE communication overhead**: Expert-parallel training/inference introduces inter-GPU communication costs not present in dense models.
5. **Mamba's selective state is lossy**: The fixed-size recurrent state cannot perfectly preserve all context information — for tasks requiring precise recall over very long contexts, this is a fundamental limitation relative to full attention.
6. **No positional encoding validated only empirically**: The claim that no positional encoding is needed is based on Mamba's implicit handling; this may break down in edge cases.

---

## Open Questions

1. **Optimal ratio at larger scales**: Is 1:7 attention-to-Mamba the right ratio at 100B+ parameters? Does the optimal ratio shift as model size grows, or as context length requirements change?

2. **Can Mamba layers be further specialized?**: The paper uses standard Mamba uniformly. Could different Mamba variants (e.g., with different state sizes) be used at different depths to better handle short vs. long-range dependencies?

3. **Instruction tuning dynamics**: Do hybrid SSM-attention models require different RLHF/instruction tuning recipes compared to pure Transformers? The recurrent state introduces non-standard behavior for dialogue/multi-turn tasks.

---

## Related Concepts

[[State Space Model|State Space Models]]
[[Selective State Spaces]]
[[Mixture of Experts]]
[[KV Cache Optimization]]
[[KV Cache Optimization|Long-Context Modeling]]
[[Efficient Transformer Architectures]]
[[GQA|Grouped Query Attention]]
[[RMSNorm]]
[[SwiGLU]]
[[Sliding window attention|Sparse Attention]]
[[LSTM|Recurrent Neural Networks]]
[[BPE Tokenization]]
