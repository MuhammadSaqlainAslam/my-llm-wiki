---
title: "Muon Optimizer"
tags: [optimization, training, deepseek, gradient, convergence]
aliases: [Muon, Muon optimizer]
tldr: "Instead of Adam's element-wise second-moment scaling, orthogonalize the gradient matrix before applying it — project it onto the nearest orthogonal matrix via Newton-Schulz iterations. The orthogonalized update is more isotropic across the weight matrix, adapts better to ill-conditioned loss landscapes, and converges faster than AdamW without needing its per-parameter statistics."
theme: efficiency
---

# Muon Optimizer

> Used in [[DeepSeek_V4]], 2026. Based on Jordan et al. 2024 and Liu et al. 2025.

## Why Adam Falls Short for Matrix Weights

Adam computes per-parameter moving averages of the gradient ($\hat{m}$) and its square ($\hat{v}$), then updates each weight as:

$$W \leftarrow W - \eta \cdot \hat{m} / (\sqrt{\hat{v}} + \epsilon)$$

The division by $\sqrt{\hat{v}}$ normalizes each scalar parameter independently — it's *coordinate-wise* scaling. For a weight matrix $W \in \mathbb{R}^{n \times m}$, this treats every entry in isolation, ignoring the matrix structure. The loss landscape for matrix weights is typically ill-conditioned: gradient magnitude varies wildly across directions. Adam adapts to this per-coordinate, but the optimal update lives in the space of *matrices*, not independent scalars.

**Muon's insight:** the best matrix update — in the steepest descent sense — is the polar factor of the gradient: $G \cdot (G^T G)^{-1/2} = UV^T$ (from the SVD $G = U\Sigma V^T$). This is an orthogonal matrix. Applying it as the update step is equivalent to taking a unit-spectral-norm step in the direction of the gradient, which is invariant to the singular value distribution of $G$ — perfectly isotropic.

---

## Newton-Schulz Orthogonalization

Computing the exact SVD at every step is too expensive. Newton-Schulz iterations approximate $UV^T$ cheaply via a polynomial recurrence:

$$M_k = a M_{k-1} + b (M_{k-1} M_{k-1}^T) M_{k-1} + c (M_{k-1} M_{k-1}^T)^2 M_{k-1}$$

Each iteration is 2 matrix multiplications — no SVD required. Starting from $M_0 = G / \|G\|_F$ (normalized to ensure singular values ≤ 1), the iterations drive all singular values toward 1.

### DeepSeek-V4's Hybrid Schedule

10 total iterations, split into two phases:

| Phase | Iterations | Coefficients $(a, b, c)$ | Goal |
|---|---|---|---|
| Aggressive | 8 | $(3.4445,\ -4.7750,\ 2.0315)$ | Rapid convergence to near-orthogonal |
| Stabilizing | 2 | $(2,\ -1.5,\ 0.5)$ | Lock singular values precisely at 1 |

The two-phase schedule is faster than running a single coefficient set for all 10 steps.

---

## Full Algorithm

```
For each training step:
  G = ∇_W L               # compute gradient
  M = μM + G              # momentum buffer (Nesterov: use μM + G not M)
  O' = HybridNewtonSchulz(μM + G)   # orthogonalize
  O = O' · sqrt(max(n,m)) · γ       # rescale RMS to match AdamW scale
  W = W · (1 - ηλ) - η · O         # weight decay + update
```

The RMS rescaling factor $\sqrt{\max(n,m)} \cdot \gamma$ lets AdamW hyperparameters be reused directly — no separate Muon-specific tuning needed.

---

## Where Muon Is and Isn't Used

Muon is applied to most linear layers (attention projections, FFN weights). AdamW is kept for:
- **Embedding and prediction head** — these are lookup tables, not matrices used in matrix multiplication in the usual sense
- **RMSNorm weights** — scalar parameters, no matrix structure
- **mHC static biases and gating factors** — small scalar parameters where orthogonalization doesn't apply

---

## Key Benefits

- **Faster convergence:** the isotropic update escapes poorly-conditioned regions faster than Adam's coordinate-wise scaling
- **No second-moment accumulation:** Muon stores only a momentum buffer $M$, not a per-parameter $\hat{v}$ — slightly lower memory than Adam
- **Reuses AdamW hyperparameters:** the RMS rescaling trick means no new hyperparameter search

---

## Where It Appears

- **[[DeepSeek_V4]]** — used for the majority of model parameters; AdamW retained only for embeddings, norms, and mHC scalars

---

## Related Concepts

- [[Manifold-Constrained Hyper-Connections]] — the other training-stability innovation in [[DeepSeek_V4]]
- [[DeepSeek_V4]] — the first large-scale production model to use Muon
- [[Transformer]] — the weight matrices Muon is updating
