---
created: "2026-08-14"
title: "Improving Language Models by Retrieving from Trillions of Tokens"
authors: "Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George van den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, Diego de Las Casas, Aurelia Guy, Jacob Menick, Roman Ring, Tom Hennigan, Saffron Huang, Loren Maggiore, Chris Jones, Albin Cassirer, Andy Brock, Michela Paganini, Geoffrey Irving, Oriol Vinyals, Simon Osindero, Karen Simonyan, Jack W. Rae, Erich Elsen, Laurent Sifre"
year: 2021
arxiv: "2112.04426"
tags: [retrieval, scaling-laws, non-parametric-memory, chunked-cross-attention, efficiency]
citation_count: 0
tldr: "RETRO augments a Transformer with a frozen, 2-trillion-token retrieval database: for every 64-token chunk of input, it retrieves the k nearest-neighbor chunks (via frozen BERT embeddings + approximate nearest-neighbor search) and lets the decoder cross-attend to them through dedicated Chunked Cross-Attention layers. A 7.5B RETRO model matches GPT-3/Jurassic-1 on the Pile with 25x fewer parameters, showing retrieval can substitute for a large chunk of parametric memorization."
aliases: ["RETRO", "Retrieval-Enhanced Transformer"]
---

# Improving Language Models by Retrieving from Trillions of Tokens

> Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, et al. (DeepMind), "Improving Language Models by Retrieving from Trillions of Tokens" (arXiv:2112.04426, ICML 2022)

## TL;DR

Scaling a language model's *parameters* is one way to make it know more facts — but it's an expensive way. Most of what a huge model learns during pretraining is just memorizing text it's already seen, baked permanently into its weights. RETRO asks: what if the model could look facts up in an external database instead of memorizing them?

RETRO (Retrieval-Enhanced TRansfOrmer) attaches a frozen, 2-trillion-token retrieval database to a fairly ordinary Transformer decoder. As the model processes text in small chunks, it retrieves the nearest-neighbor chunks from that database and cross-attends to them at every layer via a dedicated **Chunked Cross-Attention (CCA)** mechanism. The headline result: a 7.5B-parameter RETRO model matches GPT-3 (175B) and Jurassic-1 on the Pile — with **25x fewer parameters** — because it's retrieving facts instead of memorizing them.

## The Problem / Motivation

Parametric LLMs store everything they know — grammar, reasoning patterns, *and* facts — in the same set of weights, learned via gradient descent over the training corpus. This conflates two very different things: general language ability (genuinely needs capacity) and specific factual knowledge (in principle, could just be looked up). Scaling parameters to memorize more facts is expensive at both training time (more FLOPs) and inference time (bigger model to serve), and updating a fact later means retraining. Could a model instead keep a small "reasoning" core and offload most of its factual knowledge to an external, frozen, searchable database?

## The Idea

Concrete example: suppose the model is generating text that starts quoting *Hamlet*: "To be, or not to—". A purely parametric 7B model may or may not have memorized the exact continuation. RETRO instead retrieves the nearest-neighbor chunks to what it's just generated from its retrieval database — and if the database contains a copy of *Hamlet*, the retrieved chunk literally contains "not to be, that is the question," which the model can then directly attend to and copy from, rather than guess from a fuzzy memorized trace. The same mechanism works for any local, chunk-level factual continuation, not just quotations — a chunk about "the capital of Australia" pulls in a database chunk that says "Canberra."

## Architecture / Method

1. **Chunking.** The input sequence is split into contiguous chunks of a fixed size (64 tokens in the paper). A toy example from the paper: a sequence of length 12 split into 3 chunks of 4 tokens, retrieving `k=2` neighbors of 5 tokens each per chunk.
2. **Retrieval database.** Built from MassiveText (over 5 trillion tokens); RETRO's main experiments retrieve from a 2-trillion-token subset. Each database entry is a key-value pair: the key is a **frozen, pretrained BERT** embedding of a chunk, the value is that chunk plus its continuation from the source document.
3. **Nearest-neighbor search.** For each input chunk, its BERT embedding is used to query the database via **approximate nearest-neighbor search (ScaNN)**, retrieving the k nearest chunks (plus their continuations). The BERT retriever is frozen specifically so the database's embeddings never need to be recomputed during training.
4. **Chunked Cross-Attention (CCA).** Retrieved neighbor chunks are encoded by a small bidirectional encoder, then the main decoder cross-attends to them through CCA layers — interleaved with normal self-attention blocks starting from a middle layer (every third block from block 9 onward, i.e. 9, 12, 15, …). Causality is carefully preserved: retrieved neighbors for chunk *u* can only influence chunk *u*'s *later* tokens and chunk *u+1* — no information from the future leaks backward.
5. **Training.** Only the small bidirectional encoder, the CCA layers, and the decoder are trained; the retrieval database and the BERT retriever are entirely frozen and untouched by gradients. RETRO can be trained from scratch, or an existing pretrained Transformer can be cheaply "RETROfitted" with CCA layers added afterward.

```
input chunk₁  input chunk₂  input chunk₃  ...
     │              │              │
     ▼              ▼              ▼
 BERT embed → ScaNN k-NN search over 2T-token frozen DB
     │              │              │
     ▼              ▼              ▼
 retrieved      retrieved      retrieved
 neighbors₁     neighbors₂     neighbors₃
     │              │              │
     ▼              ▼              ▼
   [bidirectional encoder] → CCA layers (causally masked)
     │              │              │
     ▼              ▼              ▼
        Main decoder (interleaved self-attn + CCA)
                     │
                     ▼
              next-token predictions
```

## Key Results

- With a 2-trillion-token retrieval database, a 7.5B-parameter RETRO matches GPT-3 (175B) and Jurassic-1 on the Pile — **25x fewer parameters** for comparable perplexity.
- State-of-the-art perplexity on Wikitext103 (3.92) when retrieving from the full MassiveText database.
- Retrieval provides a roughly *constant* gain across model scales from 150M to 7B parameters, and gains grow further simply by enlarging the database or the number of retrieved neighbors at *evaluation* time, without retraining.
- Trained with only 2 neighbors, but performance keeps improving up to 10–40 neighbors at inference — larger models exploit more neighbors better (7B benefits up to ~40; 172M plateaus around 10).
- Qualitatively, retrieval measurably reduces hallucination; RETRO can recognize a snippet it's been prompted with (e.g., the opening of *Hamlet*) and generate the correct continuation by directly leveraging the retrieved match.
- The paper introduces an evaluation methodology aware of train/test proximity, to distinguish genuine reasoning from simple neighbor-copying — and shows RETRO's gains come from *both*, not just copying.

## Comparison to Prior Work

- vs. **[[RAG]]** — RAG retrieves once per query at the input/output level (a handful of passages feeding a BART generator); RETRO retrieves *per 64-token chunk* throughout generation and integrates retrieval at every layer via CCA, at a retrieval-database scale (trillions of tokens) orders of magnitude larger than RAG's Wikipedia-scale corpus.
- vs. pure parametric scaling (GPT-3-style) — RETRO shows that a fraction of the parameters, plus a frozen external database, can match a much larger purely-parametric model's factual performance, decoupling "capacity for reasoning" from "capacity for memorization."
- vs. REALM-style retrieval pretraining — REALM integrates retrieval only for masked-LM pretraining objectives at a smaller scale; RETRO targets general autoregressive language modeling at trillion-token database scale with a dedicated cross-attention mechanism for using retrieved chunks throughout generation.

## Limitations

- The BERT retriever is frozen — it's never fine-tuned end-to-end with the rest of the model, so retrieval quality is bounded by off-the-shelf BERT embeddings' notion of similarity.
- Building and serving a multi-trillion-token approximate nearest-neighbor index is a major infrastructure investment, separate from the model itself.
- Retrieval only helps when the needed information is actually present in the database; it doesn't grant novel reasoning ability, just cheaper access to memorized-elsewhere facts.
- Evaluation validity is subtle: because the model has (indirect) access to the training corpus at inference via retrieval, naive test-set perplexity can be inflated by train/test overlap — the paper's leakage-aware evaluation methodology exists specifically to address this, and any RETRO-style system needs to account for it.

## Why It Matters

RETRO was one of the first papers to show that retrieval-augmentation scales to trillions of tokens and can trade off against raw parameter count on genuinely competitive language modeling benchmarks — not just narrow QA. It directly influenced the modern retrieval-augmented production stack (RAG-style pipelines, retrieval-augmented pretraining, and later architectures that bake retrieval deeper into the model), establishing that "look it up" can be a serious alternative to "memorize it" at frontier scale.

## Related Concepts

[[RAG]] · [[Self-RAG]] · [[RAPTOR]] · [[Transformer]] · [[KV Cache Optimization]]
