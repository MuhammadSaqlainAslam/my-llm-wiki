---
created: "2026-10-19"
title: "Titans: Learning to Memorize at Test Time"
authors: "Ali Behrouz, Peilin Zhong, Vahab Mirrokni (Google Research)"
year: 2025
arxiv: "2501.00663"
citation_count: 23
tags: [ssm, memory, attention, long-context, test-time-training, recurrence, architecture, efficiency]
tldr: "A deep neural long-term memory that *learns what to store at test time* via surprise-driven gradient updates (momentum + forget gate), paired with short-term attention and persistent task vectors. Three fusion variants (MAC/MAG/MAL) beat Transformers and modern linear RNNs on LM, recall, genomics, and time series — and scale past 2M context where full attention cannot."
aliases: [Titans, Learning to Memorize at Test Time, LMM, Titan]
theme: efficiency
---

# Titans: Learning to Memorize at Test Time

> Ali Behrouz, Peilin Zhong, Vahab Mirrokni (Google Research), "Titans: Learning to Memorize at Test Time", 2025 (arXiv:2501.00663)

## The Problem / Motivation

Every sequence model is a **memory system**. The wiki’s usual framing works here too:

| Family | Memory | Write | Failure |
|---|---|---|---|
| [[Transformer\|Transformers]] | growing KV (no compression) | append | $O(n^2)$, fixed window |
| Linear RNNs / SSMs ([[Mamba]], [[RWKV]], [[RetNet]], [[Griffin]]) | fixed vector/matrix state | compress blindly | overflow / overwrite |
| Linear attention / DeltaNet | matrix state | outer-product or delta rule | still one flat store |

The contradiction: linear models are sold for *very long* context, but a tiny state cannot hold a million tokens of structure. And almost every architecture treats memory as a single process — humans don’t (short-term vs long-term vs task knowledge).

Titans asks: what if **long-term memory is a neural net whose weights are written at test time**, with a real forgetting rule, while attention only handles the recent past?

## The Idea

Three cooperating stores:

1. **Long-term memory** $\mathcal{M}$ — an MLP whose *parameters* are updated online on the input stream (meta in-context learner).
2. **Short-term memory** — causal attention over a limited window (precise local dependencies).
3. **Persistent memory** — a few learned, *data-independent* vectors prepended as task prior (not overwritten by the stream).

Writes are driven by **surprise**: if the memory can’t predict this token’s value from its key, it updates hard. If the stream changes topic, it can **forget**. Retrieval is just a forward pass — no weight update.

## Architecture / Method

### Surprise-based write (the core update)

Project $x_t$ to key/value $(k_t, v_t)$. Associative loss:

$$
\ell(\mathcal{M}_{t-1}; x_t) = \|\mathcal{M}_{t-1}(k_t) - v_t\|_2^2
$$

Momentary surprise is the gradient. Naive SGD-on-the-stream misses “a whole episode after one shock,” so they use **momentum** (past surprise) + **data-dependent rates**:

$$
S_t = \eta_t S_{t-1} - \theta_t \nabla \ell(\mathcal{M}_{t-1}; x_t)
$$

$$
\mathcal{M}_t = (1-\alpha_t)\,\mathcal{M}_{t-1} + S_t
$$

- $\eta_t$ — how much recent surprise carries over (context stickiness)
- $\theta_t$ — how much to write now
- $\alpha_t$ — **forget gate** ($0$ keep history, $1$ wipe) — i.e. adaptive weight decay

Read: $y_t = \mathcal{M}^*(q_t)$ (forward **without** the update).

This is **GD + momentum + weight decay** as an RNN write rule. Setting $\eta_t=0$ recovers [[Gated-DeltaNet-2|Gated DeltaNet]]-style updates; Titans adds momentum, forget, and **deep** MLP memory (not just one matrix).

### Parallel / matmul-friendly training

Chunk the sequence; rewrite mini-batch GD with data-dependent $\theta,\alpha$ as matmuls + a parallel associative scan on $S_t$ (same trick class as TTT). Memory training is no longer a sequential Python loop — it can sit inside a normal Transformer step.

### Three ways to fuse (Titans variants)

| Variant | Where $\mathcal{M}$ sits | Intuition |
|---|---|---|
| **MAC** — Memory as Context | long past as extra context for attention | retrieve then attend locally |
| **MAG** — Gated Memory | parallel branch, gated mix | like NSA’s multi-branch mix, but memory is a trained MLP |
| **MAL** — Memory as a Layer | deep stack of memory layers | pure recurrent depth of fast weights |

Plus **persistent memory** tokens at the front so the model still has a stable “what task is this” signal after the stream has overwritten most of the long-term store.

## Key Results

| Axis | Result (paper) |
|---|---|
| LM + commonsense (Wikitext, LAMBADA, PIQA, HellaSwag, …) | Titans **>** modern linear RNNs and their sliding-window hybrids |
| vs full Transformers | competitive / better at **same** window; scales where they OOM |
| Context | **> 2M** tokens with strong needle-in-haystack |
| BABILong, DNA, time series | wins over Transformer and linear baselines |
| Deep memory | MLP memory **beats** matrix-only (supports their “is deep memory needed?” claim) |
| Training | parallelizable (matmul + scan); not a slow online-only learner |

## Comparison to Prior Work

- vs **[[Mamba]] / [[Griffin]] / [[xLSTM]]** — those compress into fixed state with *learned* gates in the forward pass. Titans’ state is **written by an inner optimization** on associative recall loss, with explicit surprise.
- vs **DeltaNet / TTT-layers** — same “fast weights” family. Titans generalizes Gated DeltaNet and Longhorn (momentum, forget, deep memory) and TTT (forget + momentum; deep memory actually evaluated).
- vs **[[Attention\|softmax attention]]** — attention is *accurate* short-term; Titans treats it as short-term and pushes history into $\mathcal{M}$ so the window can stay small.
- vs **[[RetNet]] / gated retention** — retention is a fixed-form recurrence; Titans’ write is gradient-based meta-learning.
- vs **[[Native Sparse Attention|NSA]]** — both cut long-context cost with “compress + select + local,” but NSA compresses *tokens for attention*; Titans stores history in *weights of a second network*.

## Limitations

- Inner-loop training at test time is more moving parts than a pure forward pass (even if parallelized).
- “Surprise = gradient of $\|M(k)-v\|^2$” assumes associative key–value structure; other objectives may be better for some domains.
- Reported models are research scale (not a 100B+ production report); systems cost of deep fast-weights still needs kernels.
- Persistent memory is tiny — task prior capacity is limited.
- Hyperparameters ($\eta,\theta,\alpha$ nets) add another design surface.

## Why It Matters

This is the cleanest modern statement of the wiki’s memory thesis: **attention = short-term, trained-at-test-time neural memory = long-term, learned vectors = task prior**. It unifies linear RNNs / DeltaNet / TTT as special cases of one write rule and then *beats* them by taking forgetting, momentum, and depth seriously. For anyone building long-context systems (or reading [[Mamba]], [[Matrix memory]], [[Recurrent-Depth-Reasoning]]), Titans is the reference “what is memory, really?” paper of 2025.

## Related Concepts

[[Transformer]] · [[Attention]] · [[Mamba]] · [[Mamba3]] · [[RWKV]] · [[RetNet]] · [[Griffin]] · [[xLSTM]] · [[Matrix memory]] · [[Covariance update rule]] · [[Gated-DeltaNet-2]] · [[Recurrent state]] · [[Memory hierarchy]] · [[KV Cache]] · [[Long Range Arena]] · [[Recurrent-Depth-Reasoning]] · [[Test-Time-Compute]] · [[State Space Model]] · [[Linear attention]]