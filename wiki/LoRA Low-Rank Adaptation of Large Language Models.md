---
created: "2026-06-10"
title: "LoRA: Low-Rank Adaptation of Large Language Models"
authors: Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen
year: 2021
arxiv: 2106.09685
tags: [foundational, fine-tuning, inference, attention, parameter-efficient]
citation_count: 20561
tldr: LoRA freezes pretrained weights and injects trainable low-rank matrix pairs into Transformer layers, reducing GPT-3 175B trainable parameters by 10,000× and GPU memory by 3× with no added inference latency.
---

## The Problem

Full fine-tuning works great but doesn't scale. When GPT-3 has 175 billion parameters, producing a separate fine-tuned copy for every downstream task means storing and serving 175B floats per task — completely impractical in production. You can't just have twenty different 350GB model checkpoints sitting around for twenty different customers.

Prior attempts at parameter-efficient adaptation had their own problems. **Adapter layers** add small bottleneck modules between Transformer blocks. They're tiny in parameter count, but they add *sequential* computation that can't be parallelized away. At batch size 1 (typical online inference), adapters add 20–30% latency on GPT-2 medium — and that gets worse at scale when model parallelism requires expensive cross-GPU synchronization through the extra depth. **Prefix tuning** avoids extra layers but instead burns some of your sequence length on learnable "soft prompt" tokens, leaving less room for actual content, and is notoriously hard to optimize stably.

So the field was stuck in a triangle: full fine-tuning quality, adapter efficiency, or prefix tuning flexibility — but you had to sacrifice at least one corner. The core unsolved challenge was getting parameter efficiency *without paying for it at inference time*.

## The Idea

**The weight changes needed for adaptation live in a tiny subspace.** During fine-tuning, you're updating a weight matrix W ∈ ℝ^(d×d), but the actual update ΔW doesn't need to use all d² degrees of freedom — it's intrinsically low-rank. So instead of learning ΔW directly, factor it as two small matrices: ΔW = BA, where B ∈ ℝ^(d×r) and A ∈ ℝ^(r×d), with rank r ≪ d.

The beautiful consequence: at inference time you just add BA back into W (W' = W + BA), collapse everything into a single matrix, and the model is architecturally identical to the original. Zero extra latency, zero extra depth. The adapter tax disappears entirely.

## How It Works

**Setup:** Take any pretrained weight matrix W₀ (e.g., the query projection Wq in a self-attention layer). Freeze it completely — no gradients flow through it. Inject a parallel path with two new matrices: A initialized from a Gaussian, B initialized to zero. This ensures ΔW = BA = 0 at the start of training, so you smoothly continue from the pretrained model's behavior.

**Forward pass:** The output becomes:
```
h = W₀x + (BA)x = W₀x + ΔWx
```
You scale the BA product by α/r where α is a constant hyperparameter — this is a learning-rate-like rescaling that makes hyperparameter search more stable across different rank choices.

**Training:** Only A and B have gradients. For GPT-3 175B, if you apply LoRA to the query and value projection matrices with rank r = 4, the trainable parameter count drops from 175B to roughly 4.7M — about 0.01% of the original. The optimizer (Adam) only needs to maintain momentum and variance estimates for these tiny matrices, which is where the 3× memory reduction comes from.

**Deployment:** Compute W' = W₀ + BA once, save it, serve it. The deployed model has the *exact same* architecture and parameter count as the original. Task-switching means swapping out different (Bᵢ, Aᵢ) pairs on top of the shared frozen backbone — fast and cheap.

**Where to apply it:** The paper focuses on the attention weight matrices (Wq, Wk, Wv, Wo) in each Transformer layer. Empirically, spreading a fixed parameter budget across Wq and Wv (rather than just one of them) at a lower rank tends to work best. The rank r can be as small as 1 or 2 and still perform well — even when the full matrix dimension d is 12,288.

The key theoretical motivation comes from prior work showing that over-parameterized models have low "intrinsic dimensionality" — the loss landscape of fine-tuning effectively lives in a much lower-dimensional space than the number of parameters suggests. LoRA operationalizes this directly.

## Key Results

- **Trainable parameters:** 10,000× fewer than full fine-tuning on GPT-3 175B (≈350MB vs ≈700GB of parameter deltas)
- **Memory:** 3× reduction in GPU memory with Adam optimizer (no optimizer states for frozen weights)
- **Inference latency:** 0% overhead vs full fine-tuning (adapters added 20–30% latency at batch size 1)
- **Quality on GPT-3 (WikiSQL NL2SQL):** LoRA matches or exceeds full fine-tuning (74.0% vs 73.8% accuracy)
- **Quality on GPT-2 (E2E NLG):** LoRA matches fine-tuning across BLEU, NIST, METEOR metrics while using far fewer parameters
- **RoBERTa / DeBERTa (GLUE):** Matches or slightly exceeds fine-tuning baseline across most tasks
- **Training throughput:** Higher than fine-tuning because fewer parameters require gradient computation

## Limitations

- **Rank is a fixed hyperparameter:** LoRA commits to a rank r upfront, and different layers may benefit from different ranks. Uniform rank allocation is suboptimal — some layers may be over-parameterized while others are under-parameterized. (Later work like AdaLoRA addresses this.)
- **Which matrices to adapt:** The paper applies LoRA to attention projections and shows this works well, but there's no principled automated way to decide which weight matrices deserve LoRA modules — it's still a design choice requiring ablation.
- **Doesn't close the gap entirely in all settings:** While usually on-par, LoRA occasionally slightly underperforms full fine-tuning on specific benchmarks; the low-rank approximation is a fundamental information bottleneck.
- **Merging cost at task-switch time:** If you want to switch between tasks rapidly during serving, you need to perform the W₀ + BA merge or keep unmerged copies, adding a small bookkeeping cost.
- **Not all parameters covered:** LoRA in its basic form typically ignores MLP layers and LayerNorm parameters, which may matter for some tasks.

## Why It Matters

LoRA is arguably the most practically impactful fine-tuning paper of the LLM era. It made fine-tuning large models accessible to researchers without industrial-scale GPU clusters: fine-tuning a 7B parameter model on a single consumer GPU became possible. It directly enabled the explosion of open-source fine-tuned models (Alpaca, Vicuna, and thousands of Hugging Face community models all use LoRA or its variants).

More conceptually, it validated the **low intrinsic dimensionality** hypothesis for adaptation at scale: you really don't need to move all 175B weights to teach a model a new skill. The rank-1 or rank-2 results are almost philosophically striking — a 12,288-dimensional weight matrix can be meaningfully updated via a rank-2 perturbation.

The design's elegance — no new architecture, no inference overhead, composable with other methods — made it the default building block for parameter-efficient fine-tuning (PEFT), spawning a lineage of follow-up work: QLoRA (4-bit quantization + LoRA), AdaLoRA (adaptive rank), DoRA (decomposed direction + magnitude), and more.

## See Also

[[Transformer|Attention Is All You Need]] · [[Transformer]] · [[GPT-3]] · [[Prefix Tuning]] · [[Adapter Layers]] · [[QLoRA]] · [[Parameter-Efficient Fine-Tuning]] · [[LoRA-X]] · [[Cross-LoRA]] · [[OLoRA]]
