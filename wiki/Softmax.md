---
created: "2026-06-03"
title: "Softmax"
tags: [glossary, activation-functions]
tldr: "Converts a vector of raw logits into a probability distribution that sums to 1.0"
---

## TL;DR
Softmax exponentiates each element and normalizes by the sum, turning arbitrary real numbers into a valid probability distribution over K classes.

## Intuition
Given logits [2.0, 1.0, 0.1], softmax gives roughly [0.70, 0.26, 0.10] — the gap between classes gets *amplified* because of the exponentiation. A logit difference of 1.0 doesn't mean one class is linearly more likely; it means e^1 ≈ 2.7× more likely. This amplification is a feature: it sharpens peaked distributions and punishes confident wrong answers hard during cross-entropy training.

In the [[Attention]] mechanism, softmax sits at the heart of the score normalization step: softmax(QKᵀ/√d). Here the temperature matters enormously — without the √d scaling factor, dot products in high dimensions (e.g., d=64 → √d≈8) grow large, pushing softmax into saturation where gradients vanish to ~0. This is why scaled dot-product attention exists.

## Why It Matters
- **Attention scores**: Every transformer layer runs softmax over sequence length L; at L=128K this is 128K values per head per token — numerical stability (log-sum-exp trick) and fused kernels ([[FlashAttention]]) are non-negotiable.
- **Output logits**: Final layer softmax over vocabulary (e.g., 128K tokens for Llama 3) converts raw scores to token probabilities used in sampling and [[Speculative Decoding]] verification.
- **Training stability**: Softmax + cross-entropy loss has a clean gradient (p_i - y_i), but near-zero probabilities cause log(0) → -inf, requiring epsilon clamping or label smoothing in practice.

## Related Concepts
[[Attention]], [[FlashAttention]], [[Transformer]], [[Speculative Decoding]], [[GQA]]
