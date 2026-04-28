---
title: "Multi-Token Prediction (MTP)"
tags: [training, speculative-decoding, inference, nemotron, efficiency]
tldr: "Auxiliary heads predict 2, 3… tokens ahead simultaneously during training. Richer gradient signal improves quality ~2.4% on average; at inference the heads become draft tokens for speculative decoding at ~97% acceptance rate."
theme: synthesis
---

# Multi-Token Prediction (MTP)

Standard language model training gives you one gradient signal per token position: predict the next token, compute cross-entropy loss, backpropagate. You use each position exactly once. MTP adds auxiliary prediction heads that, during the same forward pass, predict tokens 2, 3, ... steps ahead simultaneously. Each auxiliary head takes the hidden state at position $t$ and predicts $x_{t+k}$ directly, not via the main next-token head. The model must internalize representations that make not just the immediate next token predictable but a short horizon of future tokens. This discourages lazy local-matching strategies and rewards more structured internal representations — reasoning chains, code indentation patterns, argument structure, anything that has predictable multi-step dependencies. In [[Nemotron-3]] Super, two MTP layers are added with **shared weights** — the MTP heads use the same parameters as the corresponding base model layers, so the extra parameter count is minimal. Training improvement: ~2.4% average across MMLU, MMLU-Pro, MBPP, ARC-Challenge, and GSM8K on an 8B-active MoE base model. The second benefit emerges at inference time: the MTP head's $k$-step-ahead predictions are exactly what [[Speculative Decoding]] needs for draft tokens. At ~97% acceptance rate for first-two drafts, running MTP adds almost no latency while enabling the verifier to confirm multiple tokens per forward pass.

## Where it appears

- **[[Nemotron-3]]** — architectural component (2 shared-weight MTP layers), training signal, and key enabler of throughput during RLVR rollout generation and final inference
- Not used in [[Transformer]], [[Mamba]], or [[Mixture-of-Experts]] as standalone papers — MTP is a Nemotron-3-introduced element in this wiki

## Why it matters

- **It converts a training trick into an inference optimization.** Most training-time improvements don't directly accelerate inference. MTP is unusual: the auxiliary heads learned during training become draft token generators at inference with no additional model weights, no separate draft model to deploy, no warm-up cost.
- **The training signal is richer per FLOP.** A standard next-token loss gives you one learning signal per position. With 2 MTP heads you get three overlapping signals at almost the same forward pass cost (the extra compute for 2 auxiliary heads is small relative to the full model). The model sees more of the consequence of its representations before backprop.
- **At long generation lengths it compounds.** [[Speculative Decoding]] acceptance rate of 97% on first-two drafts means roughly 2× token throughput in the best case. Over a 64K-token generation (as in [[RLVR]] rollouts), this doubles the effective number of rollouts the training loop can sample per hour, directly accelerating RL convergence.

---

*Related: [[Nemotron-3]] · [[Speculative Decoding]] · [[RLVR]]*
