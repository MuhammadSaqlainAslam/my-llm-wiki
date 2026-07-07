---
created: "2026-07-06"
title: "GLM-5: From Vibe Coding to Agentic Engineering"
authors: "GLM-5 Team (Zhipu AI & Tsinghua University)"
year: "2026"
arxiv: "2602.15763"
tags: [model-family, moe, agentic, coding, zhipu]
tldr: "Zhipu AI and Tsinghua's next-generation MoE foundation model, introducing Dynamic Sparse Architecture (DSA) for reduced training/inference cost, state-of-the-art agentic coding, and a post-training approach via asynchronous RL that decouples generation from training"
citation_count: 0
---

## TL;DR

GLM-5 advances the General Language Model line with Dynamic Sparse Architecture (DSA), a new MoE design that reduces training and inference cost while maintaining long-context fidelity. Post-trained via an asynchronous RL infrastructure that decouples token generation from weight updates, achieving state-of-the-art on real-world coding tasks and agentic benchmarks. The family spans GLM-5 (base), GLM-5.1 (agentic RL post-training, 754B parameters), and GLM-5.2 (1M context, coding-focused).

---

## Architecture

- Mixture-of-Experts with Dynamic Sparse Architecture (DSA — Dense-Sparse-Alternating) that reduces all-to-all communication overhead
- GLM-5.1 variant: 754B total parameters, per-token active parameters not disclosed
- GLM-5.2 variant: same MoE+DSA base, 1M token context window for repository-scale coding tasks
- MIT license on open weights — fully commercial-use permitted

---

## Training

- Asynchronous RL infrastructure: decouples generation and training to avoid GPU idle time during rollout collection, following the same direction as [[Multi-Environment RLVR Training]] in NVIDIA's Nemotron 3 Super pipeline
- Post-training specialized for long-horizon agentic RL, avoiding early plateaus via two-stage pipeline design

---

## Why It Matters

- DSA is a distinct MoE efficiency approach, different from [[LatentMoE]] (NVIDIA's token-compression-before-routing) — both address MoE inference cost from different angles
- MIT-licensed 754B model is directly competitive with closed and partially-open models at similar scale — relevant for organizations wanting frontier capability without usage restrictions
- The "vibe coding → agentic engineering" framing captures a real trend: moving from autocomplete-style assistance to genuinely autonomous end-to-end task completion, the same paradigm shift motivating [[MAI-Thinking-1]] and [[Kimi K2]]

---

## Limitations

- Per-token active parameter count and expert configuration not itemized in the published model card — limits precise FLOPs comparison with other MoE models
- Agentic benchmark results are first-party; independent leaderboard reproduction for SWE-Bench Pro, Terminal-Bench 2.0, and BrowseComp results pending

---

## Related Concepts

*Lineage: [[DeepSeek-V3 Technical Report|DeepSeek-V3]] · [[Kimi K2]] · [[Qwen3 Technical Report|Qwen3]] · [[LatentMoE]] · [[Multi-Environment RLVR Training]] · [[MAI-Thinking-1]]*
