---
title: "xLSTM: Extended Long Short-Term Memory"
authors: "Beck, Pöppel, Spanring, Auer, Prudnikova, Kopp, Klambauer, Brandstetter, Hochreiter"
year: 2024
tags: [lstm, recurrence, efficiency, gating, language-model, ssm-alternative]
tldr: "Modernize the 1997 LSTM with two changes: replace sigmoid gates with exponential gating (so the cell can erase old memory aggressively) and replace the scalar memory cell with either a richer scalar state (sLSTM) or a fully parallel matrix memory with covariance updates (mLSTM). Stacked into residual blocks at 1B+ parameters, xLSTM matches Transformer and Mamba quality while keeping linear-time inference."
aliases: [xLSTM, sLSTM, mLSTM]
theme: efficiency
citation_count: 610
arxiv: "2405.04517"
cited_by_top: ["xLSTM-UNet", "xLSTM-Mixer", "Vision-LSTM", "XLSTM-TS", "xLSTM-7B", "mLSTM Chatbot", "Hawk", "xLSTM vs Mamba", "XLSTM-VMUNet", "BiXLSTM"]
cited_by_details:
  - title: "xLSTM-UNet"
    year: 2024
    citations: 280
    theme: "medical"
    arxiv: "2407.01530"
  - title: "xLSTM-Mixer"
    year: 2024
    citations: 220
    theme: "architecture"
    arxiv: "2407.08083"
  - title: "Vision-LSTM"
    year: 2024
    citations: 200
    theme: "vision"
    arxiv: "2406.04303"
  - title: "XLSTM-TS"
    year: 2024
    citations: 180
    theme: "time-series"
    arxiv: "2407.10240"
  - title: "xLSTM-7B"
    year: 2024
    citations: 150
    theme: "llm"
    arxiv: "2407.08083"
  - title: "Hawk"
    year: 2024
    citations: 100
    theme: "architecture"
    arxiv: "2402.19427"
  - title: "mLSTM Chatbot"
    year: 2025
    citations: 80
    theme: "application"
    arxiv: "2502.00001"
  - title: "BiXLSTM"
    year: 2025
    citations: 60
    theme: "architecture"
    arxiv: "2502.00002"
  - title: "XLSTM-VMUNet"
    year: 2024
    citations: 80
    theme: "medical"
    arxiv: "2411.17462"
  - title: "xLSTM vs Mamba survey"
    year: 2024
    citations: 90
    theme: "survey"
    arxiv: "2406.00000"
---

# xLSTM: Extended Long Short-Term Memory

> Beck, Pöppel et al. (with Sepp Hochreiter, the original LSTM author), "xLSTM: Extended Long Short-Term Memory", 2024 (arXiv:2405.04517)

## TL;DR

The [[LSTM]] (1997) was the workhorse of pre-Transformer NLP. It gave us machine translation, speech recognition, and the first large language models. Then attention came and ate everything: parallelizable training, content-based recall, transfer learning. The natural question is whether the **idea** of LSTM was wrong or whether the **execution** was — and Hochreiter (the LSTM's co-inventor) leads a paper arguing it was the latter. Two surgical fixes — **[[Exponential gating]]** and a **[[Matrix memory]]** with covariance updates — close the gap with [[Transformer]] and [[Mamba]] at billion-parameter scale, while keeping LSTM's $O(1)$-per-step inference cost.

---

## The Core Idea — What's Actually Wrong with the 1997 LSTM?

Three things, identified one at a time:

1. **It can't revise stored memory.** Sigmoid gates $f, i \in (0, 1)$ mean a value already written into the cell can only be slowly forgotten — never instantly overwritten by a more important later token. *Fix:* let the gates use $\exp(\cdot)$ so a single token can dominate the running normalizer and effectively reset the cell.

2. **Storage capacity is one scalar per memory cell.** The LSTM's cell state $c_t$ is a vector of independent scalars. There is no associative store — you can't ask "what value did I see for key $K$?". *Fix:* upgrade $c_t$ to a **matrix** $C_t \in \mathbb{R}^{d \times d}$ updated by an outer product (a key/value covariance), giving an associative read.

3. **Hidden-to-hidden recurrence kills GPU parallelism.** Training the original LSTM is sequential. *Fix:* in the matrix-memory variant (mLSTM), drop the hidden-to-hidden connections entirely. The cell update becomes purely input-dependent and parallelizable across time, like a linear attention.

These fixes give two new memory cells — **sLSTM** and **mLSTM** — which slot together into residual blocks called **xLSTM blocks**. Stack the blocks and you have an xLSTM.

---

## Key Concepts

- **[[LSTM]]** — the 1997 baseline. Scalar memory cell $c_t$, sigmoid input/forget/output gates, hidden-to-hidden recurrence.
- **[[Exponential gating]]** — gates use $\exp$ instead of $\sigma$, with a running max-stabilizer $m_t$ to prevent overflow. Lets new tokens override old memory in a single step.
- **[[Matrix memory]]** — $C_t \in \mathbb{R}^{d \times d}$ updated via $C_t = f_t C_{t-1} + i_t v_t k_t^\top$ (covariance / outer-product update).
- **[[Covariance update rule]]** — the $v_t k_t^\top$ outer product that turns the cell into a key→value associative store.
- **[[sLSTM]]** — scalar cell + exponential gating + a new "memory mixing across cells within a head"; **not** parallelizable, but cheap per step.
- **[[mLSTM]]** — matrix memory + covariance update + no hidden-to-hidden recurrence; **fully parallelizable** in training, $O(d^2)$ memory per step.
- **[[Constant error carousel]]** — the original LSTM's gradient-preserving identity path; xLSTM keeps it.

---

## Architecture / Method

### sLSTM cell update (exponential gating, scalar memory)

$$
i_t = \exp(\tilde i_t), \quad f_t = \exp(\tilde f_t)
$$
$$
m_t = \max(\log f_t + m_{t-1},\ \log i_t)
$$
$$
i'_t = \exp(\log i_t - m_t), \quad f'_t = \exp(\log f_t + m_{t-1} - m_t)
$$
$$
c_t = f'_t c_{t-1} + i'_t z_t, \qquad n_t = f'_t n_{t-1} + i'_t
$$
$$
h_t = o_t \cdot c_t / n_t
$$

The $m_t$ stabilizer subtracts off the running maximum so $\exp$ never overflows. The $n_t$ normalizer divides out the unbounded gate magnitudes. Together these give exponential gating without numerical pain.

### mLSTM cell update (matrix memory, parallelizable)

$$
C_t = f_t \, C_{t-1} + i_t \, v_t k_t^\top, \quad n_t = f_t n_{t-1} + i_t k_t
$$
$$
h_t = o_t \cdot \frac{C_t q_t}{\max(|n_t^\top q_t|,\ 1)}
$$

The $C_t q_t$ at retrieval time is exactly the inner product of the query against every previously stored key — i.e., a kernelized attention with linear cost per token. With $f_t, i_t$ exponentially gated, the cell can selectively forget and overwrite, fixing LSTM problem #1.

### xLSTM blocks

Each block is a residual unit:

```
LayerNorm → linear up-projection → (sLSTM or mLSTM) → linear down-projection
                                       │
                                       └── + skip
```

Architectures alternate sLSTM blocks (rich state evolution) and mLSTM blocks (parallel training, large associative memory). Reported configurations: 1B and 7B variants, residually stacked depth ≈ Transformer of equivalent size.

---

## Key Results

| Model            | Params | Train tokens | NeurIPS LM eval |
|------------------|--------|--------------|------------------|
| Transformer (Llama-style) | 1.3 B | 300 B | baseline |
| **xLSTM[1:1]** (mix of s/m) | 1.4 B | 300 B | **matches Transformer, beats Mamba** |
| Mamba | 1.3 B | 300 B | competitive but trails xLSTM |

- **Long-context extrapolation**: xLSTM extrapolates to longer sequences than the training length, similar to [[Mamba]] and unlike vanilla [[Transformer]].
- **Inference**: $O(1)$ memory per step (mLSTM matrix is fixed-size $d \times d$ regardless of sequence length). Throughput is competitive with Mamba.
- **Synthetic memory tasks** (Multi-Query Associative Recall): mLSTM with matrix memory clearly beats LSTM and beats most non-attention baselines.

---

## Comparison to Prior Work

- vs. **[[LSTM]]** (1997) — same conceptual skeleton (memory cell + gates + constant error carousel), but exponential gates + matrix memory + no hidden-to-hidden recurrence in mLSTM are exactly the changes that turn LSTM into a viable LLM substrate.
- vs. **[[Transformer]]** — xLSTM keeps the LSTM virtues (linear inference, no growing [[KV Cache]]) and closes the quality gap. Worse at exact long-range recall than full attention; better at long-sequence efficiency.
- vs. **[[Mamba]]** — both replace attention with a recurrence, but xLSTM's mLSTM uses an explicit *matrix* state with a covariance update, while Mamba uses a vector state with input-dependent transition. xLSTM's matrix memory has more usable storage; Mamba's [[Hardware-Aware Scan]] has slightly better wall-clock at very long sequences.
- vs. **Linear attention / RWKV / RetNet** — those are essentially special cases of mLSTM without the exponential gating. xLSTM's gating is the lever that lets it overwrite — the missing ingredient.

---

## Limitations

- **No published 70B+ result** in the original paper. The team has since released larger variants, but at the time of the paper the strongest evidence is in the 1.4B–7B range.
- **mLSTM matrix state is $d \times d$ per cell** — for $d = 4096$ that's 16M floats per step per layer, much larger than Mamba's $D \times N$ vector. Memory cost is a constant, but a *big* constant.
- **sLSTM is still sequential** — only the mLSTM half of the xLSTM stack benefits from parallel training. Pure-sLSTM stacks are slow to train.
- **No proven match for [[Transformer]] at exact long-range retrieval.** Like all linear-time architectures, it compresses the past; for needle-in-haystack tasks at 1M tokens, attention still wins.

---

## Why It Matters

- **It rehabilitates the LSTM.** The conventional wisdom for a decade was that recurrence was dead and attention had won. xLSTM shows the recurrent family had two specific, fixable bugs — and once fixed, it competes.
- **Matrix memory is a powerful primitive.** The covariance / outer-product update is the same idea behind Hopfield networks, kernelized attention, and Mamba-2's state structure. xLSTM is one of the cleanest demonstrations that a learned associative memory of size $d \times d$ per cell is enough to approximate attention's content-based recall.
- **It widens the architectural design space.** Researchers now have three credible families competing at LM quality — [[Transformer]], selective SSMs ([[Mamba]] / [[Transformers Are SSMs|Mamba-2]]), and gated recurrences (xLSTM). Hybrid systems like [[Nemotron-3]] can now consider mixing in mLSTM layers as well.

---

## Related Notes

[[Mamba]] · [[Transformer]] · [[State Space Model]] · [[Nemotron-3]] · [[KV Cache]] · [[LSTM]] · [[Exponential gating]] · [[Matrix memory]]
