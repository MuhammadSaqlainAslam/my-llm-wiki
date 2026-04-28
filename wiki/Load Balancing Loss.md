---
title: "Load Balancing Loss"
tags: [moe, training, routing, regularization]
tldr: "Auxiliary loss that penalizes unequal token distribution across experts, preventing router collapse. Gradients flow through the differentiable softmax probability P_i even though the routing decision f_i is a hard argmax."
theme: scaling
---

# Load Balancing Loss

[[Mixture-of-Experts]] routers have a **collapse problem**: if you initialize randomly and let the router learn freely, it quickly discovers that some experts are slightly better than others and starts sending all tokens to them. Those experts improve; the others starve of gradient signal and stay bad. The feedback loop reinforces until routing is effectively non-sparse — you have $N$ experts but use 1 or 2. The standard fix (from Switch Transformers) is an auxiliary loss term:

$$\mathcal{L}_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

where:
- $f_i$ = fraction of tokens dispatched to expert $i$ in this batch (computed via hard argmax — **not differentiable**)
- $P_i$ = mean router probability assigned to expert $i$ (computed via softmax — **differentiable**)
- $N$ = number of experts; the $N$ factor keeps $\mathcal{L}_{\text{aux}}$ scale-invariant as $N$ grows
- $\alpha = 10^{-2}$ — small enough not to overwhelm the main task loss

The loss is minimized when $f_i = P_i = 1/N$ for all $i$ — perfectly uniform distribution. Gradients flow through $P_i$ (softmax output is smooth and differentiable), nudging the router probabilities toward balance even though the dispatch decision $f_i$ is a discrete argmax that can't be directly differentiated. The two quantities are coupled: if $P_i$ is high, the router will send more tokens there, so making $P_i$ uniform tends to make $f_i$ uniform over time. [[Nemotron-3]] uses a different approach for its sigmoid router: **auxiliary-loss-free balancing** — a running bias term is updated at rate $10^{-3}$ to directly correct per-expert load imbalances, with only a light auxiliary loss coefficient of $10^{-4}$ as a secondary signal. This avoids the tension between the task loss and the balancing term.

## Where it appears

- **[[Mixture-of-Experts]]** — introduced in Switch Transformers; used in Mixtral and most MoE models
- **[[Nemotron-3]]** — uses a modified auxiliary-loss-free variant with per-expert bias updates for the sigmoid router

## Why it matters

- **Without it, MoE collapses to a dense model.** Router collapse is not a theoretical concern — it happens reliably if you don't add regularization. You'd end up paying the full MoE memory cost (all $N$ expert weight matrices) while getting the compute of a dense model (all tokens hitting 1–2 experts).
- **The differentiability trick is widely applicable.** The pattern — pair a non-differentiable hard decision ($f_i$, argmax) with a differentiable soft proxy ($P_i$, softmax) in the loss — shows up in many discrete optimization problems in deep learning. Understanding it here gives you the template.
- **It's in tension with task performance.** A perfectly uniform router might not be the best router — some tokens genuinely need different processing. Tuning $\alpha$ too high homogenizes routing too much and hurts quality. The auxiliary-loss-free approach in [[Nemotron-3]] sidesteps this tension by keeping the task gradient clean and managing balance separately.

---

*Related: [[Mixture-of-Experts]] · [[LatentMoE]] · [[Nemotron-3]]*
