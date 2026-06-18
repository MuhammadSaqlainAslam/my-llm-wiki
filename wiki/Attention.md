---
created: "2026-05-28"
title: "Attention"
tags: [glossary, transformers, architecture]
tldr: "A mechanism that lets every token in a sequence dynamically weight how much it 'looks at' every other token when computing its output representation."
---

## TL;DR
Every token asks a query, every token offers a key and value — dot the query against all keys to get weights, use those weights to mix the values. That's it.

## Intuition
Think of attention as a soft, differentiable dictionary lookup. You have a query vector **q** and a bank of (key, value) pairs. Instead of hard-matching like a hash table, you compute similarity scores between **q** and every key **k**, normalize them into a probability distribution via softmax, then return a weighted sum of the values. The network learns what queries, keys, and values to produce — so it learns *what to pay attention to*, end-to-end.

The magic number is `d_k`, the key dimension (typically 64 in GPT-2, 128 in GPT-3). Scores are scaled by `1/√d_k` before softmax to prevent the dot products from growing so large that softmax saturates into near-one-hot distributions, which would kill gradients. In a 175B GPT-3, you have 96 heads × 96 layers all running this lookup in parallel, giving the model enormous capacity to route information across the sequence.

## Why It Matters
- **Context aggregation**: Unlike RNNs, attention connects any two positions in O(1) steps, letting the model resolve long-range dependencies (e.g., pronoun–antecedent agreement 100 tokens apart) without vanishing gradients.
- **Interpretability handle**: Attention weights are the most-studied proxy for "what the model is looking at" — though they are not reliably causal explanations, they remain a useful diagnostic tool.
- **Quadratic bottleneck**: Naive attention is O(n²) in sequence length — the central scaling problem driving sparse attention, linear attention, and flash attention research.

## Key Formula or Mechanism

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

```python
# shapes: Q,K,V ~ (batch, heads, seq_len, d_k)
scores = (Q @ K.transpose(-2, -1)) / d_k**0.5  # (batch, heads, seq, seq)
if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)
weights = scores.softmax(dim=-1)
out = weights @ V  # (batch, heads, seq, d_k)
```

## Where It Appears
- **Attention Is All You Need** (Vaswani et al., 2017) — original formulation
- **GPT-2 / GPT-3** (Radford et al., 2019; Brown et al., 2020) — scaled causal (masked) self-attention
- **BERT** (Devlin et al., 2019) — bidirectional self-attention for encoders
- **FlashAttention** (Dao et al., 2022) — IO-aware exact attention, same math, radically faster

## Related Concepts
[[Multi-Head Attention]]
[[Attention|Scaled Dot-Product]]
[[KV Cache]]
[[Positional Encoding]]
[[Softmax]]
