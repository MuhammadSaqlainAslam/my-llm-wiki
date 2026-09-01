---
created: "2026-09-01"
title: "Routing-Free Mixture-of-Experts"
authors: "Yilun Liu, Jinru Han, Sikuan Yan, Volker Tresp, Yunpu Ma"
year: 2026
arxiv: "2604.00801"
tags: [moe, routing, gradient-flow, expert-activation, sparse, load-balancing]
citation_count: 0
tldr: "Eliminates centralized MoE routing entirely — no external router, no Softmax, no Top-K — by letting each expert decide its own activation via a ReLU-style gate optimized through continuous gradient flow. Pairs this with a unified load-balancing framework that interpolates between expert-choice and token-choice balancing, and outperforms standard MoE, AoE, and ReMoE on language modeling."
aliases: ["Routing-Free MoE", "RFMoE"]
---

# Routing-Free Mixture-of-Experts

> Yilun Liu, Jinru Han, Sikuan Yan, Volker Tresp, Yunpu Ma (LMU Munich / MCML / UCLA), "Routing-Free Mixture-of-Experts", April 2026 (arXiv:2604.00801)

## The Problem / Motivation

Standard [[Mixture-of-Experts]] models rely on a centralized routing mechanism: a small learned router network scores every token against every expert, a Top-K (or Softmax-weighted Top-K) function picks which experts actually process the token, and a separate auxiliary loss term nudges the router toward balanced expert usage. This works, but it bakes in a rigid inductive bias: the router is a single, separately-parameterized bottleneck standing between the token and the experts that might process it, and the discreteness of Top-K makes the whole selection step awkward to optimize directly (much of the MoE literature, including the load-balancing-loss work already documented in this wiki, exists to patch around exactly this awkwardness).

## The Idea

What if there simply is no router? Instead of a centralized network scoring tokens against experts, let **each expert decide its own activation independently**, using a ReLU-style gate that's part of the expert itself and trained end-to-end through ordinary continuous gradient flow — no discrete Top-K selection step, no separate router parameters, no Softmax normalization across experts.

This builds directly on **ReMoE**'s idea of replacing Top-K + Softmax gating with a simpler ReLU-based activation decision, and extends it into a fully routing-free design: Routing-Free MoE (RFMoE) removes *all* hard-coded centralized routing machinery — external router, Softmax, Top-K, and the load-balancing loss that normally corrects for their imbalance — and encapsulates activation decisions entirely inside each expert.

## Architecture / Method

```
Standard MoE                         Routing-Free MoE (RFMoE)
────────────                         ─────────────────────────
   token x                              token x
      │                                    │
      ▼                                    ├──────────┬──────────┬── ...
┌─────────────┐                            ▼          ▼          ▼
│  Router     │  scores all experts   ┌─────────┐┌─────────┐┌─────────┐
│  (Softmax + │─────┐                 │ Expert 1││ Expert 2││ Expert 3│
│   Top-K)    │     │                 │ own ReLU││ own ReLU││ own ReLU│
└─────────────┘     │                 │ gate    ││ gate    ││ gate    │
      │             ▼                 │ decides ││ decides ││ decides │
      │      selected experts only    │ its own ││ its own ││ its own │
      ▼             run               │ activ.  ││ activ.  ││ activ.  │
  Top-K experts run                   └─────────┘└─────────┘└─────────┘
                                            │          │          │
                                            └── each expert's own gradient-flow-trained
                                                gate decides whether it activates ──┘
```

Because there's no discrete router decision to route gradients through, the whole activation decision is differentiable end-to-end — the ReLU gate inside each expert is optimized by continuous gradient flow just like any other network weight, rather than needing a separate discrete-routing training trick (Gumbel-Softmax, straight-through estimators, or a Top-K auxiliary loss).

To keep compute balanced without a centralized load-balancing loss, RFMoE adds a **unified adaptive load-balancing framework** that simultaneously optimizes both expert-balancing and token-balancing objectives through a configurable interpolation — explicitly unifying **Expert Choice (EC)** routing (each expert picks its favorite tokens, guaranteeing balanced expert load) and **Token Choice (TC)** routing (each token picks its favorite experts, the more common paradigm but prone to imbalance) as two ends of a single tunable spectrum, rather than treating them as separate design choices.

## Key Results

| Comparison | Result |
|---|---|
| RFMoE vs. standard (router-based) MoE | Consistently outperforms in language modeling |
| RFMoE vs. AoE | Outperforms |
| RFMoE vs. ReMoE | Outperforms |
| Setup | All models trained on OpenWebText under identical conditions, best-performing configurations compared |
| Additional property | Better scalability and robustness across the tested configurations |

## Comparison to Prior Work

- vs. **standard Top-K + Softmax routing** (as in [[Mixture-of-Experts]] / Switch Transformers / Mixtral) — RFMoE removes the centralized router and the discrete Top-K step entirely, letting gradient flow do what a router + auxiliary loss used to do.
- vs. **ReMoE** — RFMoE's core per-expert ReLU gating mechanism builds directly on ReMoE's replacement of Top-K/Softmax with ReLU-based gating; RFMoE's contribution on top is going fully routing-free (no centralized router of any kind) and adding the unified EC/TC-interpolating balance framework.
- vs. **Expert Choice vs. Token Choice routing** as separate paradigms — most MoE systems commit to one or the other; RFMoE's balance framework explicitly interpolates between them, which the [[MoE-Evolution-Survey|MoE Evolution survey]] elsewhere in this wiki would classify as a joint Routing + Balance innovation.

## Limitations

- Evaluated on OpenWebText-scale language modeling — it's not yet demonstrated at the scale of frontier production MoE models (hundreds of billions of parameters, trillions of tokens) where centralized routing's systems-level properties (predictable expert placement for parallelism, capacity factors for memory planning) have historically mattered as much as modeling quality.
- Removing the centralized router also removes the single place where routing behavior can be easily inspected/audited — per-expert self-activation may be harder to interpret or debug than a router's explicit token-to-expert assignment matrix.
- Comparison baselines (standard MoE, AoE, ReMoE) are the natural ones, but the paper doesn't yet report against the very latest large-scale production routing schemes (e.g., loss-free balancing as used in DeepSeek-V3).

## Why It Matters

Nearly every MoE architecture this wiki documents — [[Mixture-of-Experts]], [[LatentMoE]] — treats the router as a load-bearing, separately-designed component, and a large fraction of MoE research effort (captured in the [[MoE-Evolution-Survey|MoE Evolution survey]]) goes into patching the router's discreteness and imbalance problems after the fact. Routing-Free MoE questions that premise directly: if you can get an expert to decide its own activation via ordinary gradient flow, you don't need to solve the discrete-routing problem at all — you sidestep it. Whether or not routing-free designs eventually displace router-based MoE at frontier scale, the paper is a clean existence proof that "MoE" doesn't have to mean "centralized router," which the [[MoE-Evolution-Survey|MoE Evolution survey]]'s four-control-plane framework already anticipates as a limiting case worth tracking.

## Related Concepts

[[Mixture-of-Experts]] · [[Load Balancing Loss]] · [[LatentMoE]] · [[MoE-Evolution-Survey|The Evolution of Mixture-of-Experts Architectures]]
