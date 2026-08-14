---
created: "2026-06-10"
title: "In-Context Learning"
year: "2020"
tags: [few-shot, prompting, gpt3, language-models]
tldr: "Ability of language models to perform new tasks from examples in the prompt context — no weight updates; emerges at scale and is the basis of few-shot prompting."
aliases: ["In-Context-Learning"]
---

## Stub

> This is a concept stub. See the notes that reference this page for context.

Ability of language models to perform new tasks from examples in the prompt context — no weight updates; emerges at scale and is the basis of few-shot prompting.

## Where It Appears in This Wiki

- [[FLAN]] — contrasts trained-in zero-shot instruction-following (via finetuning) with pure in-context few-shot learning (no extra training)
- [[RAG]] — a complementary way to inject external, updatable knowledge that doesn't depend on the context window or model weights alone
- [[Gorilla]] — finds a few in-context examples improve API-call syntax but don't fix hallucination rate the way retrieval-aware finetuning does

## Related Concepts

[[RAG]] · [[FLAN]] · [[Self-Instruct]] · [[Transformer]]
