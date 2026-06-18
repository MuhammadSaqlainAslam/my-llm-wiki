---
created: "2026-06-11"
title: "Vision-LSTM"
aliases: ["Vision-LSTM", "xLSTM Vision"]
year: 2024
tags: [vision, lstm, xlstm, architecture, stub]
tldr: "xLSTM adapted for visual data: replaces the ViT attention backbone with xLSTM blocks, achieving competitive image classification with linear inference cost."
---

## TL;DR
Vision-LSTM applies the xLSTM architecture (with exponential gating and matrix memory) to image patches, replacing Vision Transformer self-attention. Gets comparable accuracy to ViT at O(n) cost per token during inference.

## See Also
[[xLSTM]] · [[State Space Models]] · [[Mamba]]
