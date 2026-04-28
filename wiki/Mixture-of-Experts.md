---
title: "Switch Transformers / Mixtral of Experts"
authors: "Fedus, Zoph & Shazeer / Jiang et al."
year: 2022
tags: [scaling, moe, routing, efficiency, sparse]
tldr: "Route each token to a sparse subset of expert FFNs; parameters scale cheaply while per-token compute stays constant. Switch uses top-1 routing; Mixtral uses top-2 of 8. More parameters, same FLOPs."
theme: scaling
---

# Mixture of Experts (MoE)

> Switch Transformers: Fedus, Zoph & Shazeer, JMLR 2022  
> Mixtral of Experts: Jiang et al., Mistral AI, 2024

## The Central Idea

Parameters and compute are two different axes you can scale along. A dense model ties them together: more parameters = more compute per token. MoE decouples them.

The key observation: **you don't need every parameter to be active for every token**. Different tokens need different kinds of processing. A token in a math equation needs different processing than a token in a poem. What if you had specialized subnetworks — "experts" — and routed each token to the most relevant ones?

That's MoE. You get a model with many parameters (large capacity, lots of knowledge) but the compute per token stays roughly constant (only a small fraction of parameters run on each token).

---

## The Basic Setup

In a [[Transformer]], each layer has:
- Multi-head attention (mixes information between positions)
- Feed-forward network (processes each position independently)

MoE replaces the FFN with a set of $N$ expert FFNs plus a router:

$$y = \sum_{i=0}^{N-1} G(x)_i \cdot E_i(x)$$

where $G(x)$ is the gating network and $E_i(x)$ is expert $i$'s output. Because $G(x)$ is sparse, most experts contribute zero — so only the selected experts actually run.

The router is simple: a linear layer followed by softmax over $N$ logits, then take the top-K:

$$G(x) = \text{Softmax}(\text{TopK}(x \cdot W_g))$$

where $\text{TopK}(\ell)_i = \ell_i$ if $\ell_i$ is among the top $K$ values, else $-\infty$.

---

## Switch Transformer: Top-1 Routing

The prevailing belief before Switch Transformers was that you need at least $K = 2$ experts per token to get non-trivial gradients through the router. Switch Transformers challenged this.

**The Switch routing rule:** Route each token to exactly **one** expert ($K = 1$). This is the "switch" — a hard binary switch between experts.

Benefits of $K = 1$:
1. Router compute cut in half
2. Each expert's buffer (capacity) can be halved
3. Communication cost reduced — each token goes to one device, not two
4. Routing is simpler, implementation is cleaner

And it works. Switch-Base with 128 experts and $K = 1$ achieves similar quality to MoE with $K = 2$, while being faster.

---

## Expert Capacity and Load Balancing

Here's a practical problem: tokens flow to experts in batches, and the router might send too many tokens to one expert. What then?

**Expert capacity** = $\left\lfloor \frac{\text{tokens per batch}}{\text{number of experts}} \right\rfloor \times \text{capacity factor}$

The capacity factor > 1 adds buffer. If an expert overflows, extra tokens skip the layer (passed through via residual connection). In practice, fewer than 1% of tokens get dropped with proper load balancing.

**Load balancing loss:** To discourage the router from collapsing to always using the same expert, they add an auxiliary loss:

$$\mathcal{L}_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

where:
- $f_i$ = fraction of tokens routed to expert $i$ (non-differentiable)
- $P_i$ = average router probability assigned to expert $i$ (differentiable)
- Minimum is achieved when $f_i = P_i = 1/N$ for all $i$ (uniform distribution)
- $\alpha = 10^{-2}$ is small enough not to overwhelm the main loss

The trick: $f_i$ is a hard argmax (not differentiable) but $P_i$ is a softmax probability (differentiable). The loss uses both, letting gradients flow through $P_i$ to encourage balanced routing.

---

## Training Instability and Fixes

MoE models are harder to train than dense models. The hard routing decision (one of $N$ experts, not a smooth interpolation) creates instability. Low precision (bfloat16) makes it worse.

**Fix 1: Selective precision.** Cast only the router computation to float32, keep everything else in bfloat16. The router produces dispatch/combine tensors that are cast back to bfloat16 before communication. You get float32 stability at nearly bfloat16 speed.

**Fix 2: Smaller initialization.** Draw weight matrices from $\mathcal{N}(0, \sqrt{s/n})$ with scale $s = 0.1$ (10x smaller than the default $s = 1.0$). This dramatically reduces variance in early training (Table 3: std drops from 0.68 to 0.01 in negative log perplexity after 3.5k steps).

**Fix 3: Expert dropout.** During fine-tuning on small datasets, sparse models overfit badly (they have far more parameters than their FLOP-matched dense baseline). Use standard dropout rate (0.1) on non-expert layers, but a much higher rate (0.4) at expert FFN layers. This boosts performance on GLUE, SuperGLUE, SQuAD.

---

## Switch Transformer: Scaling Results

The key claim: **parameters independently improve quality beyond what compute alone predicts**.

Scaling experiment (all models same FLOPs/token, same training time):
- Switch-Base with 64 experts achieves the same quality as T5-Base in **1/7 the training time**
- Switch-Base still outperforms T5-Large (which uses 3.5x more FLOPs/token)

This is the fundamental result. More parameters → better models, even with the same compute budget, if you use MoE to keep active parameters constant.

| Model | Steps to quality | Speed |
|---|---|---|
| T5-Base | 340K | 1600 ex/s |
| Switch-Base (128 experts) | 62.8K | 1000 ex/s |
| **Speedup** | **7x** | — |

They also pre-train up to **1 trillion parameters** by combining expert, model, and data parallelism across devices — experts are split across devices, so weights scale with the number of machines while each device holds a manageable slice.

---

## Mixtral 8x7B: MoE at Practice

Mixtral is a decoder-only MoE model from Mistral AI. Architecture: identical to Mistral 7B but each FFN layer is replaced by 8 SwiGLU experts, and each token is routed to $K = 2$ of them.

**Key numbers:**
- Total parameters: 47B
- Active parameters per token: 13B
- $N = 8$ experts, $K = 2$, so every token uses 2 experts
- Context length: 32k tokens
- Expert function: SwiGLU (not plain FFN)

The output for token $x$:
$$y = \sum_{i=0}^{7} \text{Softmax}(\text{Top2}(x \cdot W_g))_i \cdot \text{SwiGLU}_i(x)$$

**Performance:** Mixtral 8x7B outperforms LLaMA 2 70B on almost every benchmark, including math and code, while using 5x fewer active parameters per token during inference.

| Model | Active Params | MMLU | Math | Code (MBPP) |
|---|---|---|---|---|
| LLaMA 2 70B | 70B | 69.9% | 13.8% | 49.8% |
| GPT-3.5 | — | 70.0% | 57.1% | 52.2% |
| **Mixtral 8x7B** | **13B** | **70.6%** | **74.4%** | **60.7%** |

The efficiency story: at inference time, Mixtral is bandwidth-bound like all LLMs. With only 13B active params per token, at low batch sizes it runs like a 13B model (fast). The 47B total params matter for memory (you need to fit all experts), but not for compute per token.

---

## What Do Experts Actually Specialize In?

Surprising finding from the Mixtral routing analysis: **experts do NOT clearly specialize by domain or topic**. The distribution of expert assignments for ArXiv math, biology papers, GitHub code, and Wikipedia is nearly identical.

What they do specialize in: **syntax and position**. Common tokens like "self" in Python code, "Question" in English text, and indentation characters consistently get routed to the same expert, regardless of the broader domain. Consecutive tokens tend to go to the same expert — much more often than random (20-25% same first-choice expert vs. 12.5% chance).

The expert specialization is more about local token patterns (what kind of syntactic position is this?) than about high-level semantics (what topic is this about?). Especially at early and late layers where hidden states are more correlated with input/output embeddings.

---

## MoE vs. Dense: When to Use Each

**MoE wins when:**
- You're training-budget-constrained (sparse models learn faster per FLOP)
- You have many GPUs to spread experts across
- You're doing throughput-oriented inference (high batch sizes)

**Dense wins when:**
- Memory is tight (MoE requires fitting all expert weights)
- You need low-latency inference (memory bandwidth becomes the bottleneck when reading sparse expert weights)
- Fine-tuning on small datasets (MoE overfits more without careful regularization)

The modern solution is [[Nemotron-3]]'s LatentMoE: project tokens to a smaller latent space before routing, reducing both the communication cost and memory bandwidth bottleneck while allowing more experts.

---

## Distillation: Getting Dense Models from Sparse Teachers

You can recover ~30% of the sparse model's quality gain by distilling into a much smaller dense model. Reduce the sparse model size by up to 99% while preserving 30% of the quality improvement. This is a useful deployment trick: train big sparse model, distill to small dense model for production.

---

*Related: [[Transformer]] · [[Nemotron-3]] · [[Mamba]]*
