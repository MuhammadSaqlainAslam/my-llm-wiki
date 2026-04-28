---
title: "Attention Is All You Need"
authors: "Vaswani et al."
year: 2017
venue: NeurIPS
tags: [foundational, attention, architecture, parallelism]
tldr: "Self-attention replaces recurrence; any two tokens connect in one step, enabling parallel training and O(1) path length between positions."
theme: foundations
---

# The Transformer

> Vaswani et al., "Attention Is All You Need", NeurIPS 2017

## The Problem With RNNs

Before 2017, the dominant approach for sequence tasks was the RNN — process tokens one at a time, left to right, threading a hidden state through the sequence. The hidden state is a bottleneck. Token 1 has to survive all the way to token 512 to influence the output at position 512. And critically: **you can't parallelize training**. To compute step $t$ you need the output of step $t-1$.

This was the world the Transformer walked into and demolished.

The core insight: **you don't need recurrence at all**. You can compute dependencies between any two tokens directly, in a single step, using attention. The path length between any two positions drops from $O(n)$ to $O(1)$.

---

## Attention: The Core Operation

Imagine you're reading the word "it" in a sentence. To understand what "it" refers to, you need to look back at every other word and decide which one is most relevant. That's attention.

Formally, attention maps a **query** $Q$ against a set of **key-value** pairs $(K, V)$:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

The intuition:
- $Q$ is "what am I looking for?"
- $K$ is "what does each position have to offer?"
- $V$ is "what is the actual content at each position?"
- $QK^T$ gives a raw score for how much each position matches the query
- Divide by $\sqrt{d_k}$ to prevent the dot products from growing too large (they'd push softmax into near-zero gradient regions)
- Softmax turns scores into a probability distribution
- Weighted sum over $V$ gives the output

In matrix form, if you have $n$ tokens each represented as a $d_k$-dimensional vector, $QK^T$ is an $n \times n$ matrix — every token attends to every other token simultaneously. This is the $O(n^2)$ cost that would later become Mamba's main target.

---

## Multi-Head Attention

A single attention head can only attend to one "subspace" of the representation at a time. Multi-head attention runs $h$ attention operations in parallel, each with its own learned projections:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O$$

where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$.

The paper uses $h = 8$ heads, $d_{\text{model}} = 512$, so each head operates in $d_k = d_v = 512/8 = 64$ dimensions. Total compute cost is the same as single-head attention with full dimensionality — you just split across heads, each learning different aspects of the relationship.

In practice, different heads learn different things. Some track syntactic dependencies. Others handle coreference. The visualizations in the paper are striking — a head at layer 5 of 6 clearly learns to resolve "its" back to the correct noun, while other heads attend to local context.

---

## The Full Architecture

The Transformer is an encoder-decoder model. Each side is a stack of $N = 6$ identical layers.

**Encoder layer:**
1. Multi-head self-attention (all positions attend to all positions)
2. Position-wise FFN: $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$
3. Residual connection + LayerNorm around each sub-layer

**Decoder layer:**
1. Masked multi-head self-attention (can only attend to past positions — causal mask)
2. Multi-head cross-attention over encoder output
3. Position-wise FFN
4. Residual + LayerNorm throughout

The FFN hidden dimension is $d_{ff} = 2048$ — 4x the model dimension. This is where most parameters live in modern Transformers. Each token is processed independently by the FFN (no information mixing between positions here, only in attention).

---

## Positional Encoding

Self-attention is **permutation-invariant** — if you shuffle the input tokens, you get the same output (also shuffled). The model has no built-in notion of order. You have to add it explicitly.

The paper uses sinusoidal encodings:
$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$$

Each dimension oscillates at a different frequency. The intuition: for any fixed offset $k$, $PE_{pos+k}$ can be expressed as a linear function of $PE_{pos}$, so the model can learn to attend by relative position. In practice, learned positional embeddings perform almost identically (Table 3, row E in the paper) — the sinusoidal choice was made to potentially generalize to longer sequences.

---

## Training Details

The paper trains on WMT 2014 English-German (4.5M sentence pairs, ~37K shared BPE vocab) and English-French (36M sentences, 32K BPE vocab).

**Optimizer:** Adam with $\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$.

The learning rate schedule is notable:
$$lrate = d_{\text{model}}^{-0.5} \cdot \min(\text{step}^{-0.5},\ \text{step} \cdot \text{warmup\_steps}^{-1.5})$$

Linear warmup for 4000 steps, then decay as inverse square root of step number. This is now standard practice everywhere.

**Regularization:** dropout $P_{drop} = 0.1$ on sub-layer outputs, plus label smoothing $\epsilon_{ls} = 0.1$ (hurts perplexity but improves BLEU/accuracy).

**Results:**
- EN-DE: **28.4 BLEU** — beats all prior work including ensembles
- EN-FR: **41.0 BLEU** — single model SOTA, at 1/4 the training cost of the previous best
- Trained for 3.5 days on 8 NVIDIA P100 GPUs

---

## Why Self-Attention Beats Recurrence

The paper gives a clean comparison across three axes:

| Layer Type | Complexity/Layer | Sequential Ops | Max Path Length |
|---|---|---|---|
| Self-Attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |

For sequence modeling: self-attention connects any two positions in one step. Recurrence takes $n$ steps. Shorter paths = easier gradient flow = better long-range learning.

The $O(n^2)$ cost of attention is fine when $n < d$ (most practical cases). For very long sequences ($n \gg d$) you need tricks — this is the problem [[Mamba]] was built to solve.

---

## Key Numbers (Base Model)

| Hyperparameter | Value |
|---|---|
| $N$ (layers) | 6 |
| $d_{\text{model}}$ | 512 |
| $d_{ff}$ | 2048 |
| $h$ (heads) | 8 |
| $d_k = d_v$ | 64 |
| Dropout | 0.1 |
| Parameters | ~65M |

The "big" model doubles most dimensions: $d_{\text{model}} = 1024$, $d_{ff} = 4096$, $h = 16$, 213M parameters, 300K steps, 3.5 days on 8 P100s.

---

## What Came After

The Transformer became the universal backbone. GPT uses the decoder stack only (causal LM). BERT uses the encoder stack only (masked LM). T5 keeps the encoder-decoder. The FFN later got replaced by [[Mixture-of-Experts]] layers. The O(n²) attention cost led to [[Mamba]] and other subquadratic alternatives.

Every model you use today is either a Transformer or reacting to one.

---

*Related: [[Mamba]] · [[Mixture-of-Experts]] · [[Nemotron-3]]*
