---
created: "2026-07-08"
title: "Adapters Selector: Cross-domains and Multi-tasks LoRA Modules Integration Usage Method"
authors: "Yimin Tian, Bolin Zhang, Zhiying Tu, Dianhui Chu"
year: "2025"
arxiv: ""
technical_report: "https://aclanthology.org/2025.coling-main.40"
source_type: "conference_paper"
tags: [fine-tuning, lora, parameter-efficient, multi-task, routing]
tldr: "Trains a small 'middleman' selector adapter that routes inputs to the correct domain- and task-specific LoRA module at inference time, enabling effective cross-domain multi-task use of multiple specialized LoRA adapters without retraining them"
citation_count: 0
---

## TL;DR

Adapters Selector (AS) addresses a practical deployment problem: when you have multiple task- and domain-specific [[LoRA Low-Rank Adaptation of Large Language Models|LoRA]] adapters (e.g., one for medical QA, one for financial relation extraction, one for general text generation), you need a principled way to route an incoming input to the right adapter at inference time without human intervention. AS trains a small "selector" adapter using PEFT that learns to classify which task and domain an input belongs to, and routes it to the corresponding LoRA module.

## The Problem

[[LoRA Low-Rank Adaptation of Large Language Models|LoRA]] specialization works well per task, but maintaining separate inference pipelines per domain-task combination is impractical at deployment scale. Naive approaches — always use a single merged adapter, or manually decide which adapter to use at inference time — either degrade performance or require human intervention per request.

## The Idea

Three-stage framework:

**Stage 1 — Train domain-task LoRAs.** Each domain-specific, task-specific dataset is used independently to fine-tune a separate LoRA adapter, creating an "adapters index."

**Stage 2 — Train the selector.** Data from each dataset is sampled and mixed to train a compact selector adapter. The selector learns to map input content to the correct domain and task, using sentence embeddings and k-means-based distance metrics to determine which adapter is appropriate.

**Stage 3 — Integrated inference.** The selector and the full adapter pool are integrated with the base model. At inference, the selector classifies the input and routes it to the appropriate LoRA adapter automatically.

## Why It Matters

- Addresses a real deployment gap: most LoRA research focuses on training individual adapters, not on how to manage and route among many adapters at production scale
- Complements cross-model transfer approaches ([[Cross-LoRA]], [[LoRA-X]]) from a different angle — those ask "how do we move adapters across models," AS asks "how do we pick the right adapter for a given input"
- Published at COLING 2025 (main conference track) — peer-reviewed, not a preprint
- Code publicly available (github.com/tirant35/TASA)

## Limitations

- Selector training adds an additional fine-tuning stage that may not scale gracefully as the number of domain-task combinations grows large
- Selector accuracy depends on the quality of the sentence embeddings and the k-means distance metric used — may degrade on inputs that straddle multiple domains
- Validated primarily on relatively standard NLP tasks (medical QA, financial relation extraction, text generation); performance on more specialized or unusual domains is not established

## Related Concepts

[[LoRA Low-Rank Adaptation of Large Language Models|LoRA]] · [[OLoRA]] · [[Cross-LoRA]] · [[LoRA-X]]
