---
title: "Absolute Zero: Reinforced Self-play Reasoning with Zero Data"
authors: "Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Yang Yue, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, Gao Huang"
year: "2025"
arxiv: "2505.03335"
tags: [reasoning, reinforcement-learning, rlvr, self-play, zero-data]
tldr: "A new RLVR paradigm where a single model (AZR) simultaneously learns to propose tasks that maximize its own learning progress and improves by solving them — with zero external data, outperforming zero-setting models trained on tens of thousands of human-curated examples."
citation_count: 0
created: "2026-09-29"
---

## TL;DR
Absolute Zero Reasoner (AZR) introduces a self-contained learning loop: a single model plays two roles — a proposer that generates its own training tasks, and a solver that attempts them. A code executor validates proposed tasks and verifies answers, providing a ground-truth reward signal without any human-curated data. Despite training entirely without external data, AZR achieves overall state-of-the-art performance on coding and math reasoning, outperforming existing zero-setting models that rely on tens of thousands of in-domain human-curated examples.

## The Problem
Current RLVR approaches (like [[DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning|DeepSeek-R1]] and [[Kimi-k1.5|Kimi k1.5]]) rely on human-curated datasets of questions with verifiable answers. This creates a ceiling: performance is bounded by the quality and coverage of available human-designed problems. For a system to keep improving beyond human curation, it needs to generate its own training signal.

## The Idea
The Absolute Zero paradigm decouples task generation from task solving within one model. The model proposes a code reasoning task; a code executor runs it to verify the task is valid and solvable; then the same model attempts to solve it. The executor also verifies the solution, giving a clean reward signal. This creates a self-evolving curriculum: as the solver improves, it can propose harder tasks, driving continued learning without human input.

Three task types are used — **deduction** (predict output from code + input), **abduction** (infer input from code + output), and **induction** (infer code from input-output pairs) — covering a spectrum of reasoning modes.

## Why It Matters
- Proof of concept that a model can bootstrap its own reasoning curriculum without external data — a step toward self-improving systems.
- Outperforms zero-setting models that rely on tens of thousands of in-domain human-curated examples, despite using no labeled data.
- Extends the RLVR cluster ([[DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning|DeepSeek-R1]], [[VinePPO]], [[GRPO]], [[Kimi-k1.5|Kimi k1.5]]) with a distinctly different paradigm: no external data required.

## Limitations
- Demonstrated on code reasoning tasks where executors provide unambiguous verification — extension to tasks without a verifiable executor (open-ended writing, complex science) remains open.
- The proposer and solver share parameters, which may limit the diversity of proposed tasks relative to a separate proposer model.
- Curriculum dynamics are not fully understood: it is not always clear what drives task-difficulty progression during training.

## Related Concepts
[[DeepSeek-R1 Incentivizing Reasoning Capability in LLMs via Reinforcement Learning|DeepSeek-R1]] · [[VinePPO]] · [[GRPO]] · [[RLVR]] · [[Kimi-k1.5|Kimi k1.5]] · [[s1-Simple-Test-Time-Scaling|s1]] · [[Multi-Environment_RLVR_Training|Multi-Environment RLVR Training]]
