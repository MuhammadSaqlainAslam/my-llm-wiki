---
title: "MEDUSA: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"
authors: "Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, Tri Dao"
year: 2024
arxiv: "2401.10774"
citation_count: 751
tags: [speculative-decoding, inference, decoding-heads, tree-attention, throughput, draft-model-free]
tldr: "Medusa adds K extra decoding heads to an LLM that each predict one token further into the future. A tree-based attention mechanism verifies all candidate continuations in a single forward pass. No separate draft model needed. Medusa-1 achieves 2.2× speedup with no quality loss; Medusa-2 reaches 2.3–2.8× with joint fine-tuning."
aliases: [Medusa, Medusa decoding, MEDUSA]
theme: inference-optimization
---

# Medusa

> Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason Lee, Deming Chen, Tri Dao, "MEDUSA: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads", ICML 2024 (arXiv:2401.10774)

## TL;DR

[[Speculative Decoding]] is powerful but awkward: you need a separate draft model, which must be small enough to be cheap but large enough to be accurate. Managing two models in production is a pain, and finding the right draft model for each target model is hard.

**Medusa** sidesteps this by adding the draft model *inside* the target model as extra decoding heads. Each head $k$ predicts the token that is $k$ positions ahead of the current token. Multiple heads run in parallel in a single forward pass, generating a tree of candidate continuations. One more forward pass of the original model verifies the whole tree simultaneously.

The result is speculative decoding without a separate model — just a few extra linear layers and a smarter attention mask.

---

## The Core Idea — Extra Heads as a Built-In Draft Model

Standard autoregressive decoding generates one token at a time. Each step:

1. Forward pass through the full LLM → hidden state $h_t$
2. Project $h_t$ → logits → sample/argmax to get token $t+1$
3. Repeat

The bottleneck is that step 1 reads the entire model from [[HBM]] every step. The model is memory-bandwidth bound.

Medusa adds $K$ **extra linear heads** (each a 2-layer MLP) on top of the final hidden state. Head $k$ predicts token $t+k$:

$$
\text{head}_k(h_t) = W_k^{(2)} \cdot \text{GELU}(W_k^{(1)} h_t)
$$

These heads run in parallel during the forward pass — zero extra latency for the speculation.

### Tree-based verification

The $K$ heads each produce a distribution over the vocabulary. Greedily (or top-$p$) select $s$ candidates from each head. The Cartesian product of candidates forms a **candidate tree**:

```
Token t:   [A]
Token t+1: [B1, B2, B3]          (3 candidates from head 1)
Token t+2: [C1, C2, C3]          (3 candidates from head 2, per parent)
...
```

This tree has up to $s^K$ leaves. Medusa constructs a **tree attention mask** that allows the LLM to verify all paths in a single forward pass: each path is a valid causal sequence, but paths are packed into one batched computation via careful masking.

After verification, accept the longest prefix where the Medusa predictions agree with the model's verification pass.

---

## Key Concepts

- **[[Medusa heads]]** — the $K$ extra prediction heads attached to the frozen LLM's final hidden state
- **[[Tree attention]]** — a modified attention mask that encodes the candidate continuation tree; allows verifying all tree paths in a single LLM forward pass
- **[[Draft model-free speculative decoding]]** — Medusa's key property: no second model, just extra heads on the same model
- **[[Typical acceptance scheme]]** — Medusa's quality control: instead of accepting any token the head predicts, apply a threshold on the head's confidence to avoid low-confidence errors without running the full rejection sampling of standard speculative decoding
- **[[Medusa-1]]** — variant where only the Medusa heads are fine-tuned; the base model is frozen; guarantees lossless inference (same distribution as original model)
- **[[Medusa-2]]** — variant where both the heads and the base model are jointly fine-tuned; better head accuracy and higher speedup, but changes the model distribution

---

## Architecture / Method

### Training (Medusa-1)

1. Load a pretrained LLM; freeze its weights.
2. Attach $K$ two-layer MLP heads to the final hidden state layer.
3. Fine-tune only the heads on a dataset of (context, next-$K$-tokens) pairs. Loss: cross-entropy for each head's prediction.
4. The head's targets are shifted: head $k$ is trained to predict the token $k$ steps after the last context token.

### Training (Medusa-2)

Same, but jointly fine-tune heads and backbone with a combined loss. A **self-distillation** technique handles cases where no extra training data is available: use the frozen base model's outputs as supervision for the fine-tuned version.

### Inference

1. Forward pass → $h_t$ → produce $K$ candidate sets from $K$ heads
2. Build candidate tree (Cartesian product, pruned for top candidates)
3. One verification pass of the full LLM on the packed tree (tree attention mask)
4. Accept longest matching prefix; advance position by accepted count
5. Repeat

Typical: $K = 4$ heads, $s = 3$ candidates each → tree of 81 paths. Average accepted tokens per step: 2–3×.

---

## Key Results

| Setting | Speedup |
|---|---|
| Medusa-1 (frozen backbone) | **2.2×** (no quality loss) |
| Medusa-2 (joint finetuning) | **2.3–2.8×** |
| vs. standard speculative decoding | Comparable or slightly lower speedup |
| vs. EAGLE | Medusa: simpler; EAGLE: 3× (at cost of more complex architecture) |

Medusa was evaluated on Vicuna-7B, Vicuna-13B, and LLaMA-2-Chat models on MT-bench and GSM8K tasks.

The speedup comes almost entirely from accepting ~2–3 tokens per verification step instead of 1. The verification step is nearly free because reading the model from HBM dominates, not compute — so verifying K tokens costs barely more than verifying 1.

---

## Comparison to Prior Work

- vs. **[[Speculative Decoding]]** — standard speculative decoding needs a separate draft model. Medusa eliminates this by using extra heads. Medusa's speedup is similar (2–3×) but with far simpler infrastructure.
- vs. **[[EAGLE]]** — EAGLE achieves higher speedup (3–3.5×) by predicting at the *feature level* (second-to-last hidden state) rather than token logits. EAGLE is more accurate but requires more complex training. Medusa is simpler to implement and deploy.
- vs. **multi-token prediction** (GPT-4 style) — similar idea: predict multiple future tokens. Medusa adds heads to a pretrained model; multi-token prediction trains from scratch with multiple output heads.

---

## Limitations

- **Quality of head predictions depends on task.** For very long-form generation (code, math), head predictions are accurate and speedup is high. For highly creative generation with diverse outputs, head predictions miss more often.
- **Medusa-2 changes the model.** Joint finetuning means the model's output distribution shifts slightly — it's no longer exactly the original model's distribution.
- **Tree size grows exponentially.** With $K=5$ heads and $s=4$ candidates, the tree has 1024 leaves. Verification memory grows with tree size. In practice $K$ and $s$ are tuned for the GPU's memory budget.
- **Not applicable to encoder-only models.** Medusa is designed for autoregressive decoders.

---

## Why It Matters

- **It made speculative decoding accessible.** A single fine-tuned set of heads replaces the draft-model management problem. Anyone with a finetuned LLM can bolt on Medusa heads.
- **Tree attention is a useful primitive.** The idea of packing multiple candidate continuations into a single batched forward pass with a tree-structured mask has influenced other inference schemes.
- **It demonstrated that ~2.2× throughput improvement is achievable with minimal infrastructure change.** No new models to host, no model pairs to version. This is operationally significant.

---

## Related Notes

[[Speculative Decoding]] · [[EAGLE]] · [[KV Cache Optimization]] · [[Transformer]] · [[LLaMA 2]] · [[HBM]] · [[Inference optimization]]
