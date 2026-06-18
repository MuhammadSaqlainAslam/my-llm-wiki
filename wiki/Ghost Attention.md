---
created: "2026-05-20"
title: "Ghost Attention"
tags: [glossary, llama2, training, multi-turn, instruction-following, rlhf]
tldr: "A training trick in LLaMA 2-Chat for persistent multi-turn instruction following: the system message is appended to every turn during training but masked from the loss, teaching the model to behave as if the instruction is always present without needing to see it at every step."
aliases: [Ghost Attention, GAtt]
---

## TL;DR

When you tell a chatbot "always respond in French" at the start of a conversation, it tends to forget after a few turns. Ghost Attention fixes this: during fine-tuning, the system message (e.g., "respond in French") is silently prepended to every conversation turn in the training data. The model is trained to condition on this invisible instruction — and since the turns themselves don't repeat the instruction, the model learns that the instruction is always implicitly present.

## Intuition

Standard multi-turn training: the model sees `[system_msg] [turn 1] [response 1] [turn 2] [response 2] ...`. After many turns, the system message is far in the past and its attention weight has decayed. The model naturally "forgets" it.

Ghost Attention modifies the training data: duplicate the system message before every turn — `[system_msg] [turn 1] [response 1] [system_msg] [turn 2] [response 2] ...`. These ghost copies are masked from the loss (the model isn't penalized for predicting them), but they appear in the attention context so the model can attend to them.

During inference, no ghost copies are needed — the model has learned to behave as if the system message is always present, because it was always present during training.

## Why It Matters

- **It solves multi-turn instruction drift.** Without it, system-level instructions (persona, language, topic restrictions) erode after 5–10 turns.
- **It's a simple training data trick with no architectural change.** The model doesn't need extra parameters or a special memory mechanism.
- **It became a standard technique for chat model training.** Subsequent models adopted similar approaches to ensure system prompt persistence.

## Where It Appears in This Wiki

- [[LLaMA 2]] — Ghost Attention is introduced and evaluated in the Llama 2-Chat fine-tuning pipeline

## Related Concepts

[[LLaMA 2]] · [[RLHF]] · [[Transformer]] · [[KV Cache]]
