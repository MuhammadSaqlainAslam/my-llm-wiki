---
created: "2026-05-20"
title: "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"
authors: "Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang"
year: 2024
arxiv: "2401.15077"
citation_count: 474
tags: [speculative-decoding, inference, draft-model, feature-level, throughput, efficiency]
tldr: "EAGLE drafts at the second-to-last hidden layer (feature level) rather than token level, resolving the 'feature uncertainty' that makes token-level autoregression hard. A single lightweight autoregressive head predicts the next feature vector; the target model verifies. 3–3.5× speedup on LLaMA-2-Chat 70B — the best lossless speedup in the speculative decoding family."
aliases: [EAGLE, EAGLE speculative sampling]
theme: inference-optimization
---

# EAGLE

> Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang, "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty", ICML 2024 (arXiv:2401.15077)

## TL;DR

[[Speculative Decoding]] and [[Medusa]] draft at the *token level*: predict what the next word will be, then let the big model verify. The problem is that token-level prediction is hard — a single next-token distribution can have high entropy, making drafting inaccurate.

**EAGLE's insight**: predicting the next *feature vector* (the hidden state at the second-to-last layer) is much easier than predicting the next token. Feature space is smooth and predictable; token space is discrete and spiky.

EAGLE drafts in feature space: a small autoregressive head takes the current feature and the *actual next token embedding* (known one step later) as input, and predicts the next feature. This one-step-ahead token is the key trick that resolves the "feature uncertainty" — the draft model doesn't need to guess the next token from scratch; it just needs to propagate the feature given the already-known token.

Result: **3–3.5× lossless speedup** on LLaMA-2-Chat 70B, beating Medusa (~2.8×) and all other speculative methods at the same model family.

---

## The Core Idea — Draft at Feature Level, Not Token Level

### Why token-level drafting is hard

A language model's next-token distribution is often flat (many plausible next tokens). Drafting the argmax gives a poor acceptance rate because any non-argmax completion will be rejected.

### EAGLE's two observations

**Observation 1: Feature-level autoregression is more predictable.**
The second-to-last layer's hidden state $f_t$ (feature) changes smoothly across tokens. Predicting $f_{t+1}$ from $f_t$ is a regression task in a smooth space — far easier than sampling from a vocabulary distribution.

**Observation 2: One-step token advance resolves feature uncertainty.**
Even if $f_{t+1}$ depends on which token $t+1$ was sampled (creating uncertainty), we can resolve this by conditioning on the *actual* token $t+1$. At drafting time, EAGLE uses a tree of candidate tokens; for each candidate token $x_{t+1}$, it predicts $f_{t+1}$ conditioned on both $f_t$ and $x_{t+1}$'s embedding. This makes prediction much more accurate.

### The EAGLE draft model

A lightweight autoregressive model (a small transformer or MLP) that takes:

- $f_t$ — the current feature (second-to-last hidden state from the target model)
- $\text{embed}(x_{t+1})$ — the token embedding of the *next* token (advanced by one step)

And outputs:

- $\hat{f}_{t+1}$ — predicted next feature

From $\hat{f}_{t+1}$, apply the target model's LM head to get draft token probabilities.

The one-step token advance means the draft model conditions on the correct next token at each step — it's predicting features for the *token after next*, not the next token itself. This is the key shift that removes uncertainty.

---

## Key Concepts

- **[[Feature-level drafting]]** — predicting the LLM's internal hidden state (feature) rather than output token; the core innovation in EAGLE
- **[[Feature uncertainty]]** — the problem with token-level autoregression: the next hidden state is uncertain because it depends on which token was sampled; EAGLE resolves this via the one-step token advance
- **[[One-step token advance]]** — conditioning the feature prediction on the actual next token embedding (shifted by one), turning uncertain feature prediction into a well-conditioned regression
- **[[Draft model]]** — in EAGLE, a small autoregressive transformer trained to predict features; much smaller than the target model (~1 transformer layer)
- **[[Tree-based decoding]]** — EAGLE, like Medusa, builds a tree of candidate continuations and verifies them in one pass; EAGLE's tree is built in feature space
- **[[Lossless speedup]]** — the theoretical guarantee that accepted tokens follow exactly the target model's distribution (via rejection sampling or speculative sampling)

---

## Architecture / Method

### EAGLE draft model

The draft model is a single-layer autoregressive transformer with hidden dimension matching the target model's feature dimension. It takes:

```
Input:  [f_t, embed(x_{t+1})]    → LN → Transformer Layer → f̂_{t+1}
Output: logits = LM_head(f̂_{t+1})
```

This model is tiny — ~0.1% of the target model's parameters. It's trained by:
1. Running the target model on training data → collect $(f_t, x_{t+1})$ pairs
2. Training the draft model to minimize MSE between $\hat{f}_{t+1}$ and the actual $f_{t+1}$

### Inference loop

1. Target model produces $f_t$ (second-to-last hidden state) and $x_t$ (output token)
2. Draft model autoregressively proposes $K$ more tokens using the feature prediction:
   - For each proposed token, predict the next feature and next token
   - Build a candidate tree of depth $K$
3. One forward pass of the target model verifies all tree paths via tree attention
4. Accept the longest verified prefix; discard the rest
5. Advance by accepted count, repeat

---

## Key Results

| Model | EAGLE | Medusa | Lookahead | Vanilla |
|---|---|---|---|---|
| Vicuna-7B (MT-bench) | **2.9×** | 1.97× | 1.12× | 1.00× |
| Vicuna-13B | **3.1×** | 1.92× | — | 1.00× |
| LLaMA2-Chat 70B | **3.0×** | 2.13× | 1.88× | 1.00× |
| Mixtral 8x7B Instruct | ~2.7× | — | — | 1.00× |

EAGLE consistently leads all speculative methods. It achieves **doubled throughput** (tokens/second) on LLaMA-2-Chat 70B — the most widely used publicly available large model at the time.

The acceptance rate of EAGLE is high (80–90%+ on most tasks) because predicting the next feature given the correct next token is very accurate.

---

## Comparison to Prior Work

- vs. **[[Speculative Decoding]]** — standard speculative decoding needs a separate draft model of appropriate size; EAGLE's draft model is a single lightweight layer attached to the same backbone. EAGLE is significantly faster (3× vs. typically 1.5–2×).
- vs. **[[Medusa]]** — Medusa drafts at the token logit level; EAGLE drafts at the feature level. EAGLE: higher speedup (3× vs. 2.2×), requires training on target model's internal features. Medusa: simpler, works with a frozen backbone. EAGLE is the better choice when you can afford feature-level training.
- vs. **[[Lookahead Decoding]]** — Lookahead uses Jacobi iterations to generate multiple tokens; no separate model. EAGLE: higher speedup on most benchmarks.

---

## Limitations

- **Requires access to internal hidden states.** EAGLE must be trained on $f_t$ features from the target model; this requires forward passes on training data (but not backprop through the target model).
- **Feature space tied to architecture.** The draft model is specific to the target model's architecture and parameter count — you can't reuse an EAGLE head across different LLM families.
- **Still needs fine-tuning.** Unlike Medusa-1 (which freezes the backbone), EAGLE requires training the draft model from scratch for each target model.
- **Tree size vs. latency tradeoff.** Building larger trees increases the expected acceptance count but also increases memory and compute for the verification pass.

---

## Why It Matters

- **Best lossless throughput improvement in the speculative decoding family.** At the time of publication, 3× lossless speedup on a 70B model was the best result in the category.
- **Feature-level prediction is a general insight.** The observation that internal representations are smoother and more predictable than token distributions has influenced subsequent draft model designs.
- **It narrows the gap between speculative decoding research and production usability.** With a 3× improvement using only a tiny additional model, EAGLE makes speculative decoding attractive even for latency-sensitive production systems.

---

## Related Notes

[[Speculative Decoding]] · [[Medusa]] · [[KV Cache Optimization]] · [[Transformer]] · [[LLaMA 2]] · [[HBM]] · [[Inference optimization]] · [[Draft model]]
