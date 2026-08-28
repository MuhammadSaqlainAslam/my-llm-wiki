---
created: "2026-08-14"
title: "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
authors: "Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi"
year: 2023
arxiv: "2310.11511"
tags: [retrieval, rag, self-reflection, critique, adaptive-retrieval]
citation_count: 2464
tldr: "Trains a single LM to decide on its own when to retrieve and to critique both the retrieved passages and its own output, by learning to emit special reflection tokens as part of normal generation. Outperforms ChatGPT and retrieval-augmented Llama2-chat on open-domain QA, reasoning, fact verification, and long-form generation — with only a 7B/13B model."
aliases: ["Self-RAG"]
---

# Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection

> Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi, "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection", ICLR 2024 Oral (arXiv:2310.11511)

## TL;DR

[[RAG]] fixed a real problem — LLMs invent facts because their parametric memory is finite and stale — but it fixed it clumsily. Vanilla RAG retrieves a fixed number of passages for *every* query, whether or not the query needs external knowledge, and it never checks whether what it retrieved is actually relevant, or whether its own generated answer is actually supported by that retrieved text. Retrieval becomes a blind, unconditional preprocessing step bolted onto generation.

Self-RAG's fix: make retrieval and critique things the *model itself decides*, on the fly, as part of ordinary next-token generation. Train the LM to emit special **reflection tokens** interleaved with its normal output — one that says "should I retrieve right now, and how much," and others that grade the passages it retrieved and the text it just wrote. No separate critic model is needed at inference time — the same LM that generates the answer also judges it, because it learned to imitate a stronger critic (GPT-4) during training.

The result: a 7B/13B open model that outperforms ChatGPT and retrieval-augmented Llama2-chat on open-domain QA, reasoning, fact verification, and long-form biography generation — and, unusually for this kind of system, the amount of "carefulness" is a dial you can turn at inference time without retraining.

## The Problem / Motivation

Two failure modes of standard RAG:

1. **Retrieval is unconditional.** The pipeline retrieves top-$k$ passages for every input regardless of whether the query is "what's 2+2" or "list every treaty signed by X in the 1800s." Retrieving for questions that need no external knowledge can actively hurt fluency and versatility — you're forcing irrelevant context into the prompt.
2. **No quality control on what comes back.** Even when retrieval helps, nothing checks whether the retrieved passages are actually relevant to the query, or whether the model's generated sentence is actually supported by them. The model can retrieve a barely-related passage and still confidently hallucinate around it.

## The Idea

Self-RAG trains one LM, end-to-end, to produce four kinds of special tokens as part of its normal autoregressive output:

- **`Retrieve`** — should I retrieve passages right now, and if so how many? (decided per-segment, not once per query)
- **`ISREL`** (is-relevant) — is this retrieved passage actually relevant to the query?
- **`ISSUP`** (is-supported) — is the segment I just generated fully, partially, or not supported by the passage it was conditioned on?
- **`ISUSE`** (is-useful) — on a 1–5 scale, how useful/helpful is the overall response to the query, independent of whether it's fully factual?

**Concrete example.** For "What's the capital of France?" the model emits `Retrieve = No` and answers straight from parametric memory — no need to spend a retrieval call on a fact it already knows cold. For "Write a short biography of [obscure 19th-century chemist]," the model emits `Retrieve = Yes`, pulls several candidate passages, scores each with `ISREL`, generates a segment conditioned on the relevant ones, scores that segment's `ISSUP` against the passage, and repeats — retrieval and critique interleaved with generation rather than happening once, upfront, blind.

## Architecture / Method

**1. Collect reflection-token training data with a stronger critic.** GPT-4 is prompted (instructions + few-shot demonstrations, temperature 1, max 200 output tokens) to produce reflection-token labels over a training corpus — instances where GPT-4 did not follow the expected output format are discarded. This yields roughly 12.6K `Retrieve` labels, 11.2K `ISSUP` labels, 19.3K `ISREL` labels, and 3.8K `ISUSE` labels.

**2. Train a single generator LM to imitate this, end-to-end.** The base LM (e.g. Llama2-7B/13B) is fine-tuned via ordinary next-token prediction on data augmented with these reflection tokens, so it learns to naturally interleave `Retrieve`/`ISREL`/`ISSUP`/`ISUSE` tokens into its own output stream. Critically, **no separate critic model runs at inference time** — the trained generator emits its own critique tokens as it goes.

**3. Adaptive, on-demand retrieval at inference.** Generation proceeds segment by segment. At each segment boundary, the model itself decides (via the `Retrieve` token) whether external passages are needed for what comes next.

**4. Segment-level relevance filtering and support-checking.** When retrieval fires, each candidate passage gets an `ISREL` score; the model generates candidate continuations conditioned on each sufficiently-relevant passage, then scores each continuation's `ISSUP` against its source passage.

**5. Critique-weighted selection across branches.** Because multiple retrieved passages can each produce a candidate continuation, Self-RAG uses a beam-search-like procedure that scores each branch with a weighted combination of `ISREL`/`ISSUP`/`ISUSE` token probabilities ($w_{rel}$, $w_{sup}$, $w_{use}$) and keeps the best. These weights are tunable at inference time with no retraining — turning up $w_{sup}$ trades some fluency for higher citation/support precision.

```
Vanilla RAG:                         Self-RAG:
  query                                query
    │                                   │
    ▼                                   ▼
  always retrieve top-k          "Retrieve?" token
    │                                   │
    ▼                          ┌────────┴────────┐
  generate, conditioned      Yes: retrieve k     No: generate
  on all k passages          passages, score       directly from
  (no relevance check)       ISREL each             parametric memory
    │                              │
    ▼                              ▼
  done, no self-check      generate segment per relevant passage,
                            score ISSUP, pick best via
                            weighted beam search, repeat
```

## Key Results

- On the **FactScore** biography-generation benchmark, Self-RAG-7B scores **81.2**, a +3.2 point gain over retrieval-augmented Llama2-7B (78.0).
- Outperforms **ChatGPT** and **retrieval-augmented Llama2-chat** on open-domain QA (PopQA long-tail subset), reasoning (ARC-Challenge), fact verification (PubHealth), and long-form QA (ALCE-ASQA) — using only a 7B or 13B open model.
- **Ablations confirm every component matters**: removing the retriever entirely drops PopQA accuracy from 45.5% → 43.6%; removing the critic drops ALCE-ASQA str-em from 32.1% → 18.1% (a 14-point hit — the critique mechanism matters more than retrieval alone); disabling test-time retrieval on the full model drops PopQA to 24.7% (a 30% relative hit).
- **Inference-time controllability is real**: increasing the $w_{sup}$ (ISSUP) weight measurably increases citation precision, at some cost to MAUVE fluency — a genuine dial, not just a training-time hyperparameter.

## Comparison to Prior Work

- vs. **[[RAG]]** (Lewis et al.) — vanilla RAG retrieves unconditionally for every query and never checks relevance or support; Self-RAG makes both a learned, per-segment decision the model itself makes.
- vs. **[[RETRO]]** — RETRO bakes retrieval into the architecture itself (cross-attention to retrieved chunks at every step, decided at pretraining time); Self-RAG instead makes retrieval a controllable, on/off decision the model learns to make during generation, with no architectural change to attention.
- vs. **[[RAPTOR]]** — RAPTOR improves *what* gets retrieved (hierarchical tree summaries for better multi-level context); Self-RAG improves *whether and how* retrieval is used and *whether the output is trustworthy* — the two are complementary, not competing.

## Limitations

- Reflection-token training labels come from GPT-4, so the smaller trained model's judgment is only as good as what it learned to imitate from that critic — including any of the critic's own biases or blind spots.
- Inference is more complex than plain greedy decoding: segment-level generation with multi-passage branching and weighted beam search over reflection-token probabilities adds real compute overhead versus a single forward pass.
- The `ISUSE` "helpfulness" score is explicitly independent of factual correctness (following Liu et al. 2023's definition of perceived utility) — a response can score well on usefulness while still containing unsupported claims.

## Why It Matters

Self-RAG moved retrieval-augmented generation from a fixed pipeline (retrieve → stuff into context → generate) into a **learned, controllable, self-critiquing process** where the model decides when it needs help and grades its own work against what it found. That shift — retrieval as a decision the model makes, not a preprocessing step imposed on it — is a direct ancestor of later "agentic RAG" systems where an LLM actively manages its own retrieval and verification loop rather than passively consuming whatever a retriever hands it.

## Related Concepts

[[RAG]] · [[RETRO]] · [[RAPTOR]] · [[Lost-in-the-Middle|Lost in the Middle]] · [[Reflexion]]
