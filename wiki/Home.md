# LLM Wiki

Architecture notes in the style of Andrej Karpathy — intuition first, math second, no fluff.

---

## Foundations

| Article | One Line |
|---|---|
| [[Transformer]] | Self-attention replaces recurrence. Any token attends to any other in O(1). |
| [[Mamba]] | Selective state spaces. Linear-time sequence modeling that matches Transformer quality. |

## Scaling

| Article | One Line |
|---|---|
| [[Mixture-of-Experts]] | Decouple parameters from compute. Route tokens to specialized FFN subnetworks. |

## Modern Systems

| Article | One Line |
|---|---|
| [[Nemotron-3]] | Hybrid Mamba-Transformer-MoE. 3x throughput over pure Transformer MoE at same quality. |
| [[DeepSeek_V4]] | CSA + HCA compressed attention cuts KV cache 10× at 1M tokens. MoE + Muon optimizer. SOTA open model. |

---

## Concept Map

```
Transformer (2017)
│
├── Problem: O(n²) attention cost at long sequences
│   ├──► Mamba (2024): replace attention with selective SSM
│   └──► DeepSeek-V4 (2026): CSA + HCA compress KV cache 10× at 1M tokens
│
├── Problem: Parameters tied to compute
│   └──► Mixture-of-Experts: sparse routing, constant FLOPs/token
│
└── Problem: MoE communication + latency bottlenecks
    └──► Nemotron-3 LatentMoE: route in latent space
```

```
Mamba + MoE + few attention layers
= Nemotron-3 hybrid architecture
= best throughput-to-accuracy frontier (2025)

Standard attention + DeepSeekMoE + CSA/HCA + mHC + Muon
= DeepSeek-V4 (2026)
= 1M-token context at 27% of V3 FLOPs, SOTA open model

DeepSeek-V4 sub-concepts:
  [[Compressed Sparse Attention]]          compress + sparse select KV
  [[Heavily Compressed Attention]]         compress hard, dense attention
  [[Manifold-Constrained Hyper-Connections]]  stable residual highway
  [[Muon Optimizer]]                       orthogonal gradient updates
  [[On-Policy Distillation]]               unify specialists post-RL
  [[GRPO]]                                 group-relative RL, no critic
```

---

## Reading Order

If you're new: **Transformer → Mixture-of-Experts → Mamba → Nemotron-3**

If you care about efficiency: **Mamba → Mixture-of-Experts → Nemotron-3**

If you're deploying: **Nemotron-3** (it synthesizes everything)
