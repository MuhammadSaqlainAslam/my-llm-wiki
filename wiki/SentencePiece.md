---
created: "2026-06-03"
title: "SentencePiece"
tags: [glossary, tokenization]
tldr: "A language-agnostic subword tokenizer that trains directly on raw text, producing a fixed vocabulary of ~32k tokens."
---

## TL;DR
SentencePiece is an unsupervised tokenizer that learns a subword vocabulary (typically 32k–64k pieces) directly from raw Unicode text, requiring zero pre-tokenization or language-specific rules.

## Intuition
Classical tokenizers need whitespace splitting or language-specific preprocessing before learning subwords — meaning they silently bake in English-centric assumptions. SentencePiece treats the input as a raw stream of Unicode characters and runs BPE or Unigram LM directly on that stream. "Hello world" and "Helloworld" get handled identically; the model figures out boundaries. This makes it genuinely multilingual and reproducible: the same `.model` file deterministically encodes any string.

The Unigram LM variant (used in T5, LLaMA) works backwards — start with a huge candidate vocabulary (~100k), then iteratively prune tokens whose removal minimally increases corpus loss, until you hit your target size. The result is a probabilistic segmentation where you can sample multiple tokenizations of the same string, which acts as a data augmentation trick during training.

## Why It Matters
- **Portability:** A single `.model` artifact encodes/decodes without external dependencies — critical for serving LLMs at scale where reproducibility across environments matters.
- **Vocabulary efficiency:** Unigram's loss-driven pruning tends to allocate token budget more optimally than greedy BPE merge rules, compressing multilingual corpora into fewer tokens and reducing sequence length fed into the [[KV Cache]].
- **Byte fallback:** SentencePiece can emit raw byte tokens for unknown characters, ensuring the [[Transformer]] never sees an `<unk>` — every possible string is encodable, which matters for code, math, and adversarial inputs.

## Related Concepts
[[Transformer]], [[KV Cache]], [[Speculative Decoding]], [[RoPE]], [[GQA]]
