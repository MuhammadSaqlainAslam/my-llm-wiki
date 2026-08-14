---
created: "2026-08-14"
title: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
authors: "Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela"
year: 2020
arxiv: "2005.11401"
tags: [retrieval, rag, knowledge-intensive-nlp, open-domain-qa, hybrid-parametric-nonparametric]
citation_count: 0
tldr: "Combines a pretrained seq2seq generator (parametric memory) with a dense vector index of Wikipedia (non-parametric memory) retrieved at inference time. The generator conditions on retrieved passages instead of memorizing facts in its weights — so knowledge can be updated by swapping the index, not retraining the model. Set state-of-the-art on open-domain QA and produced more factual, specific text than a parametric-only baseline."
aliases: ["RAG", "Retrieval-Augmented Generation"]
---

# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

> Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", NeurIPS 2020 (arXiv:2005.11401)

## TL;DR

A pretrained language model stores facts in its weights — "parametric memory." That's a bad way to store facts: the model can't cite a source, can't be updated without retraining, and quietly hallucinates when it doesn't actually know something. RAG's fix is to stop asking the model to memorize the world and instead give it a library card: at generation time, retrieve the top-$k$ most relevant passages from a dense vector index of Wikipedia, and condition the generator on them.

Concretely: ask "who wrote *The Origin of Species*?" — a dense retriever pulls Wikipedia passages about Darwin and the book, and a BART-based generator reads those passages alongside the question to produce the answer. If tomorrow's Wikipedia gets edited, you swap the index — no retraining required.

RAG set the state of the art on three open-domain QA benchmarks at publication and, on generation tasks, produced measurably more factual and specific text than a parametric-only baseline of the same size. It's the paper that launched the retrieval-augmented paradigm now standard in production LLM systems.

## The Problem / Motivation

Large pretrained seq2seq models are good at fluent generation but store their "knowledge" implicitly, distributed across billions of weights learned during pretraining. This has three problems: (1) the model can't explain or cite where an answer came from; (2) updating knowledge means retraining or fine-tuning the whole model, which is expensive and can cause forgetting elsewhere; (3) the model's knowledge is frozen at its training cutoff and it will confidently generate wrong facts (hallucinate) rather than admit it doesn't know.

Separately, extractive QA systems (retrieve a passage, extract the exact span containing the answer) solve the citation and updatability problems, but they can only output spans that appear verbatim in a retrieved document — they can't synthesize, paraphrase, or combine information across multiple passages into a free-form answer.

RAG asks: can we get a generative model's fluency and synthesis ability *together with* a retrieval system's ability to cite, update and stay grounded in real documents?

## The Idea

Split "knowledge" into two separate systems that specialize in what they're good at:

- **Parametric memory** — a pretrained seq2seq model (BART-large) that's good at understanding language and generating fluent, coherent text.
- **Non-parametric memory** — a dense vector index over all of Wikipedia (accessed via Maximum Inner Product Search), which is easy to inspect, cite, and swap out.

At inference time, given an input query, a retriever (initialized from **DPR**, Dense Passage Retrieval) encodes the query and pulls the top-$k$ Wikipedia passages whose dense embeddings are closest to it. The generator then conditions on *both* the original query and the retrieved passages to produce its output — treating the retrieved document as a latent variable that gets marginalized out, so the whole thing is trained end-to-end without ever needing supervision on which specific document is "correct" for a given answer.

## Architecture / Method

```
query x
   │
   ▼
┌─────────────────────┐        ┌───────────────────────────┐
│  Retriever (DPR)     │──top-k─▶  Non-parametric memory:    │
│  bi-encoder, MIPS    │  docs  │  dense vector index of     │
│  over Wikipedia      │        │  ~21M 100-word Wikipedia   │
└─────────────────────┘        │  passages                  │
                                └───────────────────────────┘
   │  query x + retrieved doc z_i (for i = 1..k)
   ▼
┌─────────────────────┐
│  Generator (BART-    │
│  large seq2seq)      │
└─────────────────────┘
   │
   ▼
marginalize p(y|x) over the k retrieved documents z_i
```

The retriever and generator are both pretrained (DPR and BART respectively) and then fine-tuned jointly on the downstream task — only the query encoder is fine-tuned on the retriever side; the document encoder (and hence the Wikipedia index itself) is kept frozen to avoid the cost of periodically re-embedding the entire corpus.

Two ways to marginalize over the retrieved documents, which is the paper's key design choice:

- **RAG-Sequence** — treat the retrieved document as a single latent variable for the *whole* output sequence. Generate a complete candidate answer conditioned on each of the $k$ retrieved documents separately, then marginalize: $p(y|x) = \sum_{i=1}^k p(z_i|x)\, p(y|x, z_i)$. One document, one coherent generation.
- **RAG-Token** — let a *different* document contribute to each generated token. At every decoding step, marginalize over the $k$ documents' next-token distributions: $p(y_t|x, y_{<t}) = \sum_{i=1}^k p(z_i|x)\, p(y_t|x, z_i, y_{<t})$. This lets the model synthesize an answer that draws on evidence spread across several documents.

## Key Results

Exact Match (EM) on open-domain QA, evaluated on Natural Questions (NQ), TriviaQA (TQA, two eval settings — closed test-set / open unfiltered), WebQuestions (WQ), and CuratedTrec (CT):

| Model | NQ | TQA | WQ | CT |
|---|---|---|---|---|
| Closed-book T5-11B | 34.5 | 37.4 / 50.1 | 37.4 | — |
| DPR (extractive) | 41.5 | 56.8 | 41.1 | 50.6 |
| **RAG-Token** | 44.1 | 55.2 / 66.1 | 45.0 | 51.9 |
| **RAG-Sequence** | **44.5** | 56.8 / **68.0** | **45.2** | **52.2** |

RAG-Sequence edges out RAG-Token on most QA benchmarks — for exact-match-style short answers, committing to one coherent document tends to help. Notably, on TriviaQA — an extractive task where prior specialized approaches used custom pretraining objectives — RAG's free-form generation still beat prior extractive SOTA.

Beyond QA:
- **FEVER fact verification**: RAG reaches 72.5% 3-way classification accuracy — within 4.3 points of a heavily-engineered, pipeline-based specialist system trained with intermediate retrieval supervision, which RAG does not need.
- **Jeopardy question generation** (guess the entity from a fact, scored with Q-BLEU-1): RAG-Token beats RAG-Sequence, and both beat a BART-only baseline of the same size.
- **Human evaluation of factuality**: judges found RAG more factual than BART in 42.7% of cases vs. BART more factual in only 7.1% (both factual in 17% more); RAG was also judged more specific.
- **Retrieval quality on FEVER**: the top-retrieved document comes from a gold-evidence article 71% of the time; a gold article appears somewhere in the top 10 retrieved 90% of the time.

## Comparison to Prior Work

- vs. **closed-book parametric models (T5, BART)** — same generation fluency, but RAG grounds answers in retrieved text instead of relying purely on facts baked into weights, and beats closed-book T5-11B despite RAG's generator being far smaller.
- vs. **extractive QA / REALM** — extractive systems can only output a verbatim span from a retrieved document. RAG can synthesize an answer from a passage that contains clues about the answer without containing it verbatim, and RAG's retriever needs no specialized retrieval-supervision pretraining beyond DPR's initialization.
- vs. **DPR alone** — DPR pairs a dense retriever with a BERT-based cross-encoder re-ranker and an extractive reader. RAG shows that neither the re-ranker nor the extractive reader is necessary to reach comparable or better accuracy — retrieve, then let a generator read and produce the answer directly.

## Limitations

- The document encoder (and the index it produces) is frozen after DPR pretraining — only the query encoder is fine-tuned, so the retriever can't fully adapt its notion of "relevant" to the downstream task.
- Retrieval quality caps generation quality: if none of the top-$k$ documents contain the needed information, no amount of generator fluency recovers it.
- RAG-Token's per-token marginalization is more expressive but more expensive at inference than RAG-Sequence, since it requires combining $k$ token distributions at every decoding step.
- Non-parametric memory is Wikipedia passages only — the method doesn't address retrieval over structured data, code, or non-text modalities.

## Why It Matters

RAG is the paper that named and formalized the retrieve-then-generate pattern that is now the default architecture for grounding LLMs in external, updatable, citable knowledge — the foundation underneath essentially every production "chat with your documents" system, enterprise search assistant, and open-domain QA product built on top of an LLM. Its central bet — that you get more reliable, factual, and maintainable systems by separating "how to reason and write" (parametric) from "what is currently true" (non-parametric, swappable) — has proven durable even as the specific retriever/generator architectures have been replaced many times over.

## Related Concepts

[[In-Context Learning]] · [[Transformer]] · [[RETRO]] · [[Self-RAG]] · [[RAPTOR]]
