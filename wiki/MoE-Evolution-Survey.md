---
created: "2026-09-01"
title: "The Evolution of Mixture-of-Experts Architectures in Large Language Models: Routing, Topology, Load Balancing, and Expert Parallelism"
authors: "Jiguo Li"
year: 2026
arxiv: "2608.08650"
tags: [moe, routing, load-balancing, expert-parallelism, survey, topology]
citation_count: 0
tldr: "A synthesis survey arguing MoE's architectural evolution isn't a simple timeline — it's a dependency graph of eight milestones analyzable along four coupled control planes: Expert Topology, Routing, Balance, and Expert Parallelism. Draws on Switch Transformers, DeepSeekMoE, DeepSeek-V2/V3, Mixtral, and Qwen3 to build a unified formalization of what MoE simultaneously optimizes: capacity, quality, and system efficiency."
aliases: ["MoE Evolution Survey", "Evolution of Mixture-of-Experts Architectures"]
---

# The Evolution of Mixture-of-Experts Architectures in Large Language Models

> Jiguo Li, "The Evolution of Mixture-of-Experts Architectures in Large Language Models: Routing, Topology, Load Balancing, and Expert Parallelism", August 2026 (arXiv:2608.08650)

## TL;DR

[[Mixture-of-Experts]] models let you scale parameter count without scaling per-token compute — but "MoE" isn't one design, it's a family of decisions that have been made differently by every major model (Switch Transformers, Mixtral, DeepSeekMoE, DeepSeek-V2/V3, Qwen3) with little cross-paper consistency in terminology. This survey's core move is refusing to tell that history as a chronological list of releases. Instead it organizes MoE architectures along **five coupled dimensions** (expert granularity, expert topology, routing freedom, the scope of load balancing, execution structure) and reframes the field's actual history as a **dependency graph of eight architectural milestones** — six mainline developments plus two orthogonal branches — rather than eight successive generations. For analyzing any individual system, it proposes **four control planes**: Expert Topology (which experts exist), Routing (which experts process each token), Balance (how aggregate load is controlled), and Expert Parallelism (how the selected computation maps onto physical devices).

## The Core Idea — Four Control Planes, Not a Timeline

Most MoE surveys narrate architectural history chronologically: Switch Transformer → GShard → Mixtral → DeepSeekMoE → ... This survey argues that framing obscures the real structure, because later systems don't strictly supersede earlier ones — they make different, sometimes orthogonal choices along independent axes. The four control planes it proposes:

1. **Expert Topology** — how many experts exist, how they're organized (flat pool vs. shared + routed experts vs. hierarchical grouping), and at what granularity.
2. **Routing** — the mechanism that decides which expert(s) process a given token (Top-K token-choice, expert-choice, learned vs. heuristic).
3. **Balance** — how aggregate compute/token load is kept even across experts and devices (auxiliary losses, capacity factors, loss-free balancing).
4. **Expert Parallelism** — how the routing decision is physically executed across a distributed training/inference cluster (all-to-all communication patterns, expert placement).

Any specific MoE system can be described as a point in this four-dimensional space, which is what lets the survey compare, e.g., DeepSeek-V3's fine-grained shared-expert design against Mixtral's coarser flat pool without forcing one into a "predecessor" of the other.

## Architecture / Method — Eight Milestones as a Dependency Graph

Rather than eight sequential generations, the survey structures MoE's evolution as six mainline developments plus two orthogonal branches:

| Stage (mainline) | Key move |
|---|---|
| Dense/Soft MoE | Statistical division of labor across experts — not yet true computational sparsity |
| Sparse conditional computation | Top-K routing establishes real capacity leverage: only a subset of experts actually run per token |
| (further mainline milestones) | Increasing expert granularity, shared-expert designs, improved balance mechanisms |
| Orthogonal branch 1 | (topology-focused innovations, e.g. hierarchical/latent routing) |
| Orthogonal branch 2 | (parallelism/systems-focused innovations, e.g. communication-efficient expert placement) |

The survey's unified formalization treats MoE design as simultaneously optimizing three, sometimes competing, goals: **capacity** (more effective parameters), **quality** (task performance per unit of that capacity), and **system efficiency** (real-world throughput given communication and load-balancing overhead). Reading any individual paper's design choices through this three-goal lens — rather than treating each innovation as an isolated trick — is the survey's main pedagogical contribution.

## Comparison to Prior Work

- vs. treating MoE history as a straight timeline (as most blog-post-style overviews do) — this survey's dependency-graph framing correctly captures that, e.g., a communication-efficient parallelism scheme and a new routing-freedom mechanism can be adopted independently and combined, rather than one having to come "after" the other.
- vs. [[LatentMoE]] (already in this wiki, via [[Nemotron-3]]) — LatentMoE is a topology/parallelism innovation (route in a projected latent space to cut all-to-all communication) that this survey's four-control-plane framework would classify primarily under Expert Topology + Expert Parallelism, with Routing and Balance largely unaffected — a concrete example of the survey's claim that innovations are often orthogonal rather than sequential.
- vs. [[Routing-Free-MoE|Routing-Free Mixture-of-Experts]] (also newly added to this wiki, arXiv:2604.00801) — that paper is a Routing-plane innovation (eliminate the router entirely, let experts self-select via gradient flow) that also touches Balance (its unified adaptive load-balancing framework). It's a good stress test of this survey's taxonomy: does "no router" still fit inside a framework built around "which router mechanism"? The survey's four-plane structure accommodates it as a limiting case of the Routing dimension.

## Limitations

- Single-author survey (assisted by Codex, per the paper's own acknowledgment) — synthesis papers benefit from broader review; a single author's taxonomy choices necessarily reflect one perspective on how to carve up the field.
- As with any survey of a fast-moving area, the "eight milestones" framing is a snapshot as of August 2026 — new orthogonal branches (e.g., routing-free approaches) are already emerging and may eventually need a ninth axis rather than fitting cleanly into the existing four planes.
- The dependency-graph structure is a conceptual tool for organizing prior work, not a predictive theory — it doesn't tell you which future combination of choices will work best, only how to classify what's already been tried.

## Why It Matters

MoE terminology has been genuinely fragmented — "routing," "load balancing," and "expert parallelism" mean subtly different things across the Switch Transformer, Mixtral, and DeepSeek papers already in this wiki. This survey's four-control-plane vocabulary gives a consistent way to describe and compare any MoE system, including the ones already documented here ([[Mixture-of-Experts]], [[LatentMoE]]) and future ones. For anyone designing a new sparse architecture, it reframes the design problem correctly: you're not picking a single "MoE architecture," you're making four semi-independent decisions, and the survey's job is helping you see which combinations have already been tried and which competing goal (capacity, quality, system efficiency) each choice trades off against.

## Related Notes

[[Mixture-of-Experts]] · [[LatentMoE]] · [[Load Balancing Loss]] · [[Nemotron-3]] · [[DeepSeek_V4]] · [[Routing-Free-MoE|Routing-Free Mixture-of-Experts]]
