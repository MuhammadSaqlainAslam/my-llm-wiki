---
created: "2026-05-22"
title: minbpe
tags: [tokenization, BPE, byte-pair encoding, language model, GPT-2, GPT-4, educational]
year: 2024
tldr: Minimal, clean Python implementation of byte-level Byte Pair Encoding (BPE) tokenization — the algorithm used by GPT-2 through GPT-4 — with train/encode/decode support and a GPT-4-compatible tokenizer wrapper.
wikilinks: ["[[nanoGPT]]", "[[llm.c]]", "[[Inference optimization]]"]
---

# minbpe

**Repository:** [karpathy/minbpe](https://github.com/karpathy/minbpe)
**Author:** Andrej Karpathy
**Year:** 2024

## Overview

minbpe provides the cleanest possible reference implementation of **Byte Pair Encoding (BPE)** tokenization, the algorithm at the heart of virtually every modern large [[Transformer|language model]] tokenizer (GPT, Llama, Mistral, Claude, etc.).

BPE was introduced for NLP by Sennrich et al. (2015) and popularised for LLMs by the GPT-2 paper. It operates on UTF-8 bytes, iteratively merging the most frequent adjacent token pairs to build a vocabulary.

## Tokenizer Classes

| Class | Description |
|---|---|
| `BasicTokenizer` | Simplest BPE, runs directly on raw text |
| `RegexTokenizer` | Adds regex pre-splitting by category (letters, numbers, punctuation) — matches GPT-2/GPT-4 behaviour |
| `GPT4Tokenizer` | Thin wrapper around `RegexTokenizer` reproducing GPT-4 / tiktoken `cl100k_base` exactly |

All three implement the same three-method interface: **train**, **encode**, **decode**.

## Key Design Points

- **Byte-level:** vocabulary always starts with all 256 raw bytes; merges are learned on top
- **Special tokens:** registered explicitly via `register_special_tokens()` to avoid accidental tokenization of attacker-controlled data
- **Save/load:** `.model` file (loadable) + `.vocab` file (human-readable)
- Training ~25 seconds on an M1 MacBook for a moderate dataset

## Example

```python
from minbpe import RegexTokenizer
tok = RegexTokenizer()
tok.train(text, vocab_size=32768)
ids = tok.encode("hello world")
print(tok.decode(ids))  # "hello world"
tok.save("tok32k")
```

## GPT-4 Parity Verification

```python
from minbpe import GPT4Tokenizer
import tiktoken

text = "hello123!!!? (안녕하세요!) 😉"
enc = tiktoken.get_encoding("cl100k_base")
tok = GPT4Tokenizer()
assert enc.encode(text) == tok.encode(text)  # ✓ identical
```

## Relation to Wiki Themes

Tokenization is the first stage of any LLM pipeline and directly affects [[KV cache]] size (fewer tokens = smaller cache), [[speculative decoding]] draft quality, and vocabulary-level [[Scaling Laws|scaling]] decisions. minbpe makes the BPE mechanics transparent and hackable, complementing [[nanoGPT]] (training) and [[llm.c]] (C/CUDA inference).
