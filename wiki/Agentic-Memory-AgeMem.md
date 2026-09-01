---
created: "2026-09-01"
title: "Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents"
authors: "Yi Yu, Liuyi Yao, Yuexiang Xie, Q. Tan, Jiaqi Feng, Yaliang Li, Libing Wu"
year: 2026
arxiv: "2601.01885"
tags: [agents, memory, reinforcement-learning, grpo, long-term-memory, short-term-memory]
citation_count: 52
tldr: "Instead of bolting a heuristic memory controller onto an LLM agent, expose memory operations (store, retrieve, discard) as tool-based actions the agent's own policy chooses to invoke, and train that policy with a three-stage progressive GRPO curriculum designed to handle the sparse, delayed rewards that memory decisions produce."
aliases: ["AgeMem", "Agentic Memory"]
---

# Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for LLM Agents

> Yi Yu, Liuyi Yao, Yuexiang Xie, Q. Tan, Jiaqi Feng, Yaliang Li, Libing Wu, "Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for Large Language Model Agents", January 2026 (arXiv:2601.01885)

## The Problem / Motivation

Long-horizon LLM agent tasks routinely exceed what fits in a context window, and — as this wiki's [[LLM-Agent-Memory-Survey|LLM Agent memory survey]] catalogs — most existing solutions handle long-term memory (LTM) and short-term memory (STM) as **separate components**, each managed by its own heuristic or auxiliary controller: a fixed rule decides what gets summarized into short-term working memory, a different fixed rule decides what gets written out to a long-term store, and neither rule is *learned* — they're hand-designed policies bolted onto the agent from outside. [[Generative-Agents|Generative Agents]], for instance, uses a hand-crafted recency + importance + relevance scoring formula to decide what to retrieve; it works well, but it's not something the agent itself learned to do based on what actually helped it succeed.

## The Idea

Stop treating memory management as external plumbing. Expose memory operations — store this, retrieve that, discard this — as **tool-based actions inside the agent's own action space**, exactly like any other tool call (search, calculator, code execution). The agent's policy then decides, at every step, whether and how to use its own memory, and that decision-making is trained end-to-end with reinforcement learning rather than fixed by a human-designed heuristic. Long-term and short-term memory stop being two separately-engineered subsystems and become two kinds of actions the same learned policy can choose between.

The hard part is the reward signal: storing a piece of information now might only pay off dozens of steps later when it's retrieved and used — a **sparse, discontinuous reward** that's hard for standard policy-gradient training to credit correctly. AgeMem addresses this with a **three-stage progressive reinforcement learning strategy** built on step-wise [[GRPO]] (Group Relative Policy Optimization), designed specifically to handle this sparse/delayed credit-assignment problem for memory actions.

## Architecture / Method

```
Agent's action space (unified):
   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐
   │ environment  │  │  tool calls  │  │ STORE (LTM/  │  │ RETRIEVE     │
   │ actions      │  │  (search,    │  │  STM memory) │  │ (LTM/STM     │
   │ (task-       │  │   calc, ...) │  │              │  │  memory)     │
   │  specific)   │  │              │  │              │  │              │
   └─────────────┘  └─────────────┘  └─────────────┘  └──────────────┘
          all chosen by the SAME learned policy π, trained via GRPO

Three-stage progressive RL curriculum:
   Stage 1 ──▶ Stage 2 ──▶ Stage 3
   (simpler memory      (harder, longer-       (full long-horizon
    credit-assignment    horizon memory          task, sparse reward
    sub-tasks first)     dependencies)           end-to-end)
```

Because memory operations are just actions, the same step-wise GRPO machinery used to train an agent's task actions also trains its memory decisions — but the three-stage curriculum ramps up the difficulty of the credit-assignment problem gradually, rather than throwing the full sparse-reward long-horizon setting at the policy from the start.

## Comparison to Prior Work

- vs. **[[Generative-Agents|Generative Agents]]'s heuristic memory stream** — Generative Agents retrieves via a fixed formula (recency + importance + relevance), hand-designed and never updated by feedback about whether it actually helped the agent. AgeMem instead learns what to store/retrieve directly from task reward, via RL, and unifies LTM/STM into one policy rather than two separately-tuned subsystems.
- vs. **[[Reflexion]]'s reflection buffer** — Reflexion's "memory" is a growing list of verbal self-critiques, written and read by a fixed procedure (always reflect after failure, always prepend past reflections to the next attempt) — not something the policy decides to do or not do. AgeMem's memory actions are optional, learned choices within the agent's own action space.
- vs. **[[LLM-Agent-Memory-Survey|Memory for Autonomous LLM Agents]]**'s taxonomy — AgeMem is a clean instance of that survey's "**policy-learned management**" mechanism family (the least common of the five families the survey identifies, since most existing systems use fixed heuristics instead), applied jointly across both temporal-scope dimensions (LTM and STM) at once.

## Limitations

- Sparse, delayed rewards for memory actions are exactly the kind of signal reinforcement learning struggles with most; the three-stage curriculum is a mitigation, not an elimination of the underlying difficulty — training stability and sample efficiency likely remain harder than for standard task-action RL.
- Unifying LTM/STM into one learned policy adds training complexity (three RL stages, step-wise GRPO) compared to a simple heuristic controller that requires no training at all — a real engineering cost that only pays off if the learned policy generalizes better than hand-tuned heuristics across the tasks it's deployed on.
- As with most agent-memory papers, evaluation is necessarily on a fixed set of long-horizon benchmark tasks; how well a learned memory policy transfers to genuinely novel task distributions it wasn't trained on is an open question.

## Why It Matters

AgeMem is a concrete answer to a gap the [[LLM-Agent-Memory-Survey|LLM Agent memory survey]] explicitly flags: almost every memory mechanism in production and in the literature — including [[Generative-Agents|Generative Agents]]'s influential memory stream — uses a fixed, hand-designed control policy for write/manage/read decisions, not a learned one. AgeMem shows that memory management can be folded into the same RL training loop already used for an agent's task behavior, treating "should I remember this" as just another action worth optimizing rather than a separately-engineered subsystem. That's a meaningful step toward agents whose memory behavior improves with more training data and experience, the same way their task-completion ability does.

## Related Concepts

[[Generative-Agents|Generative Agents]] · [[Reflexion]] · [[GRPO]] · [[RLVR]] · [[LLM-Agent-Memory-Survey|Memory for Autonomous LLM Agents]]
