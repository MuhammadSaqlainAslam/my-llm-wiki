---
created: "2026-06-10"
title: "InstructGPT"
authors: "Ouyang et al."
year: "2022"
arxiv: "2203.02155"
tags: [foundational, llm, alignment, rlhf]
citation_count: 23339
tldr: "OpenAI's instruction-tuned GPT-3 using RLHF — the paper that made language models reliably follow user instructions. See the full RLHF note for the technical details."
---

InstructGPT is OpenAI's application of [[RLHF]] to GPT-3 (arxiv 2203.02155). See [[RLHF]] for the full technical breakdown of the training pipeline.

Its supervised fine-tuning stage builds on the same instruction-following goal as [[FLAN]] (2021) and [[Self-Instruct]] (2022) — the difference is that InstructGPT adds a human-preference reward model and PPO on top, where FLAN uses only supervised instruction data and Self-Instruct bootstraps that data from the model itself instead of human annotators.
