---
created: "2026-09-01"
title: "VL-JEPA: Joint Embedding Predictive Architecture for Vision-language"
authors: "Delong Chen, Mustafa Shukor, Théo Moutakanni, Willy Chung, Jade Yu, Tejaswi Kasarla, Yejin Bang, Allen Bolourchi, Yann LeCun, Pascale Fung"
year: 2025
arxiv: "2512.10942"
tags: [vision-language, jepa, self-supervised, embeddings, multimodal]
citation_count: 0
tldr: "Instead of autoregressively generating text tokens one at a time, VL-JEPA predicts continuous embeddings of the target text directly — learning in an abstract representation space that abstracts away surface-level linguistic variability. Built on a V-JEPA 2 vision encoder plus a Llama-3-initialized predictor, it beats CLIP/SigLIP2/Perception Encoder on video classification and retrieval, and matches classical autoregressive VLMs on VQA at 1.6B parameters."
aliases: ["VL-JEPA"]
---

# VL-JEPA: Joint Embedding Predictive Architecture for Vision-language

> Delong Chen, Mustafa Shukor, Théo Moutakanni, Willy Chung, Jade Yu, Tejaswi Kasarla, Yejin Bang, Allen Bolourchi, Yann LeCun, Pascale Fung (Meta FAIR, HKUST, Sorbonne Université, NYU), "VL-JEPA: Joint Embedding Predictive Architecture for Vision-language", December 2025 (arXiv:2512.10942)

## The Problem / Motivation

Classical vision-language models — [[MiniGPT-4]] and the LLaVA line ([[LLaVA-1.5|LLaVA (Improved Baselines with Visual Instruction Tuning)]]) among them — bolt a vision encoder onto a language model and train the whole thing to **autoregressively generate discrete text tokens**, one at a time, conditioned on image embeddings. This inherits everything that comes with token-level autoregressive generation: the model is forced to commit to exact surface wording, small token-level errors compound over a generated sequence, and every bit of the model's capacity spent matching the reference text's *exact phrasing* is capacity not spent on the underlying visual-semantic content the phrasing is trying to express. Two different but equally correct captions for the same image look, at the token level, like two very different targets.

## The Idea

Don't generate text tokens. **Predict a continuous embedding of the target text instead**, and only decode that embedding back into human-readable words at the very end, if a sentence is actually needed. This is the Joint Embedding Predictive Architecture (JEPA) philosophy — already applied to video-only self-supervised learning in [[V-JEPA-1|the original V-JEPA]] and [[V-JEPA-2|V-JEPA 2]], both in this wiki — extended for the first time to vision-**language**: predict in embedding space, where semantically equivalent targets (two different phrasings of the same answer) naturally end up close together, rather than in token space, where they can look arbitrarily different.

## Architecture / Method

```
       X_V (image / video frames)          X_Q (text query)
              │                                   │
              ▼                                   │
      ┌───────────────┐                           │
      │   X-Encoder    │  (V-JEPA 2 ViT)            │
      │  visual tokens │                           │
      └───────────────┘                           │
              │                                   │
              └───────────────┬───────────────────┘
                               ▼
                     ┌───────────────────┐
                     │     Predictor       │  (Llama-3 layers,
                     │  visual + query      │   query-conditioned)
                     │  tokens in, pooled    │
                     │  + projected out       │
                     └───────────────────┘
                               │
                               ▼
                        Ŝ_Y  (predicted embedding)
                               │
              ┌────────────────┴────────────────┐
              │  training: compare to             │  inference (only if text
              │  true target embedding S_Y         │  is actually needed):
              │  via InfoNCE loss (like CLIP)       │  Y-Decoder → text
              └────────────────────────────────────┘

   Y (target text)
         │
         ▼
   ┌───────────────┐
   │   Y-Encoder    │  → S_Y  (true target embedding, training only)
   └───────────────┘
```

Four components: an **X-Encoder** (a V-JEPA 2 Vision Transformer producing visual tokens), a **Y-Encoder** (maps the ground-truth textual target into an embedding $S_Y$, used only during training), a **Predictor** (initialized from Llama 3 Transformer layers, taking both the visual tokens and the tokenized textual query as input, with its output pooled and projected into the Y-Encoder's embedding space to produce $\hat{S}_Y$), and a lightweight **Y-Decoder** (translates a predicted embedding back into readable text — used only at inference time, and only when a sentence is actually required).

Training objective: $\mathcal{L}_{\text{VL-JEPA}} = D(\hat{S}_Y, S_Y)$ — distance in embedding space — instead of the classical VLM objective $\mathcal{L}_{\text{VLM}} = D(\hat{Y}, Y)$ over raw token sequences. Concretely this is an **InfoNCE loss**, the same contrastive-loss shape CLIP uses: pull the predicted embedding toward the true target embedding, push it away from other samples' target embeddings in the batch, plus a regularization term (standard in JEPA-family training) to prevent representation collapse.

## Key Results

| Comparison | Result |
|---|---|
| vs. standard token-space VLM training (same vision encoder, same data, controlled comparison) | Stronger performance with **50% fewer trainable parameters** |
| Selective decoding (skip decoding when not needed) | **2.85×** fewer decoding operations, similar performance vs. non-adaptive uniform decoding |
| Video classification (8 datasets) + video retrieval (8 datasets), average | Surpasses **CLIP, SigLIP2, and Perception Encoder** |
| VQA (GQA, TallyQA, POPE, POPEv2) | Comparable to classical autoregressive VLMs (InstructBLIP, QwenVL) |
| Parameter count | **1.6B** — smaller than the classical VLMs it matches on VQA |

The embedding space also directly supports **open-vocabulary classification** and **text-to-video retrieval** with no architecture modification — a byproduct of predicting in a shared embedding space rather than generating discrete tokens, which classical autoregressive VLMs can't do without a separate contrastive head.

## Comparison to Prior Work

- vs. **[[V-JEPA-1|V-JEPA]] / [[V-JEPA-2|V-JEPA 2]]** — those apply the joint-embedding-predictive idea to video *only* (predict masked video-embedding regions from visible ones, no language involved). VL-JEPA is the natural extension into vision-language, directly reusing a V-JEPA 2 encoder as its X-Encoder — a concrete example of the JEPA framework generalizing across modalities rather than being video-specific.
- vs. **[[MiniGPT-4]] and [[LLaVA-1.5|LLaVA]]** — both are classical autoregressive VLMs: vision encoder + language model, trained to generate text tokens directly. VL-JEPA reuses the "vision encoder + Transformer" shape (its Predictor is even initialized from Llama 3) but replaces token-level generation with embedding-level prediction, achieving comparable VQA performance at roughly a third of the parameters of typical models in that class.
- vs. **CLIP / SigLIP2 / Perception Encoder** — those are dual-encoder contrastive models good at retrieval/classification but with no generative capability at all. VL-JEPA beats them on their own retrieval/classification turf while *also* supporting text generation via its Y-Decoder — getting both capabilities from one embedding-space-trained model instead of needing separate contrastive and generative models.

## Limitations

- The Y-Decoder (embedding → readable text) is described as lightweight and used only when a sentence is actually needed — its own generation quality/fluency isn't the paper's focus, and free-form long-text generation quality may not match a model trained end-to-end for fluent generation.
- Matching (not exceeding) classical autoregressive VLMs on VQA at 1.6B params is a strong efficiency result, but the paper doesn't yet establish whether the embedding-prediction approach continues to scale favorably at the 10s-of-billions-of-parameters scale where today's frontier VLMs operate.
- Training relies on an InfoNCE-style contrastive objective, which needs a sufficiently large and diverse batch of negative samples to work well — a consideration CLIP-style models share and that can complicate scaling batch size at very large model scale.

## Why It Matters

VL-JEPA is a direct test of a bet this wiki already tracks in [[V-JEPA-1|V-JEPA]] and [[V-JEPA-2|V-JEPA 2]]: that predicting in embedding space, rather than in raw pixel or token space, is a more efficient and more semantically-grounded self-supervised objective. Extending that bet from video-only to vision-language — and getting a model that's simultaneously a strong retriever, classifier, *and* generator at a fraction of the usual parameter count — is evidence the JEPA philosophy isn't a video-specific trick but a genuinely modality-general alternative to the "encode vision, decode language tokens" recipe that has otherwise been nearly universal since [[MiniGPT-4]] and LLaVA popularized it.

## Related Concepts

[[V-JEPA-1|V-JEPA (Revisiting Feature Prediction for Learning Visual Representations from Video)]] · [[V-JEPA-2|V-JEPA 2]] · [[MiniGPT-4]] · [[LLaVA-1.5|LLaVA (Improved Baselines with Visual Instruction Tuning)]] · [[Transformer]]
