---
created: "2026-05-12"
title: "Long Short-Term Memory (LSTM)"
tags: [rnn, recurrence, gating, history, language-model]
tldr: "The 1997 Hochreiter–Schmidhuber recurrent network with a constant-error-carousel cell state and three sigmoid gates (input, forget, output). The dominant sequence model for two decades; replaced by [[Transformer]] but rehabilitated by [[xLSTM]]."
---

# Long Short-Term Memory (LSTM)

An LSTM cell maintains a scalar (or vector-of-scalars) memory $c_t$ updated as $c_t = f_t c_{t-1} + i_t z_t$, where $f_t, i_t \in (0, 1)$ are sigmoid forget and input gates. The output is $h_t = o_t \tanh(c_t)$. The crucial design is the **[[Constant error carousel]]**: when $f_t \approx 1$ and $i_t \approx 0$, the cell state passes through unchanged and gradients flow without vanishing. This solved the vanishing-gradient problem that killed vanilla RNNs and made the LSTM the workhorse of pre-attention NLP — machine translation, speech recognition, the first large language models. Two limitations doomed it relative to Transformers: (i) sigmoid gates can't override stored memory in a single step, and (ii) hidden-to-hidden recurrence prevents parallel training. [[xLSTM]] fixes both.

## Where it appears

- [[xLSTM]]
- [[Transformer]]
- [[Mamba]]

---

*Related: [[Constant error carousel]] · [[Exponential gating]]*
