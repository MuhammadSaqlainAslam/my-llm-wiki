---
title: Fixed-Point Reasoners: Stable and Adaptive Deep Looped Transformers
authors: Sajad Movahedi, Vera Milovanović, Shlomo Libo Feigin, Alexander Theus, Thomas Hofmann, Valentina Boeva, T. Konstantin Rusch, Antonio Orvieto
year: 2026
arxiv: 2606.18206
tags: [reasoning, looped-transformers, adaptive-computation, fixed-point, inference, transformer, test-time-compute]
citation_count: 0
tldr: FPRM replaces ACT-based halting in looped Transformers with fixed-point convergence detection, achieving 10+ accuracy points better than TRM on Sudoku-Extreme while using ~27% fewer effective layers of compute.
---

## The Problem

Reasoning tasks feel intuitively iterative: you try something, check if it works, refine. Looped Transformers capture this nicely — instead of stacking many different layers, you run *the same* layer repeatedly, building up depth at inference time without adding parameters. Models like HRM and TRM showed you can beat much bigger LLMs on puzzles like Sudoku, Maze, and ARC-AGI this way.

But two problems lurk under the surface. First, **when do you stop looping?** Most approaches either fix the number of iterations in advance (losing adaptivity to hard vs. easy inputs) or train an auxiliary Adaptive Computation Time (ACT) network to decide when to halt. ACT is famously hard to optimize because the halting decision is discrete, and empirically it often fails to allocate *more* compute to *harder* inputs — the whole point of adaptive inference.

Second, **deep looping creates deep networks**, and deep networks have signal propagation problems. Looped models have traditionally used post-norm layers specifically because post-norm keeps activation magnitudes bounded across iterations. But post-norm is known to cause unstable training in very deep non-looped architectures. As the number of loops grows to handle harder tasks, this tension becomes critical: you want depth for expressivity, but depth kills training stability.

## The Idea

Stop treating the halting decision as a learned module bolted on top — instead, let the loop itself tell you when it's done: **halt when the hidden state stops changing**, i.e., when it converges to a fixed-point.

If `z_{i+1} = f_θ(z_i; x)` and eventually `‖z_{i+1} − z_i‖ ≤ ε`, then the model has reached `z* = f_θ(z*; x)` — a genuine fixed-point of the recurrence. No extra module, no discrete halting probability to differentiate through: the geometry of the dynamics provides the stopping criterion for free.

## How It Works

**Fixed-point halting.** At each iteration, compute `‖z_{i+1} − z_i‖`. Once this falls below a threshold `ε`, stop. Hard inputs naturally require more iterations to converge; easy inputs converge fast. This gives adaptivity without any learned halting network. A theoretically motivated modification (dampening/anti-oscillation term) is added to prevent the iterates from perpetually bouncing around a fixed-point without converging.

**Pre-norm + residual scaling.** The key insight is: *a looped Transformer unrolled for N iterations is effectively an N-layer-deep Transformer*. Deep non-looped Transformers universally prefer pre-norm (LayerNorm before the sublayer, not after) for stable gradient flow. But post-norm has served a purpose in looped models — it keeps activation norms bounded across iterations. The fix: keep pre-norm for gradient stability, and add **learnable residual scaling parameters** (small scalars multiplying each residual branch). These damp the activation growth that pre-norm would otherwise allow, replacing the bounding role of post-norm without its training instability.

Concretely, the update looks like:
```
z_{i+1} = z_i + α · Attn(Norm(z_i)) + β · FFN(Norm(z_i + α · Attn(Norm(z_i))))
```
where α, β are learned scalars initialized small, giving the model control over how much each residual branch contributes — crucial at large effective depths.

**Architecture (FPRM).** A single Transformer block containing: ShortConv (for local context mixing, following URM), multi-head attention with NoPE (no positional encoding, avoiding RoPE artifacts in recurrence), SwiGLU FFN, all wrapped in pre-norm with residual scaling. The same block is looped until fixed-point convergence.

**No hierarchy needed.** Unlike HRM and TRM which split computation into fast-looping and slow-looping components, FPRM uses a single flat loop. The stability improvements make the single loop expressive enough.

## Key Results

- **Sudoku-Extreme:** FPRM achieves **+10 accuracy points** over TRM while using approximately **27% fewer effective layers** of compute (∼1,000 fewer layers).
- **Maze-Hard, ARC-AGI-1, state-tracking (A5 and S5):** FPRM outperforms HRM and TRM among all models at the 7M parameter scale.
- **Adaptivity:** FPRM is claimed to be the first Transformer-based reasoning model that demonstrably scales inference compute with actual input difficulty — easy Sudoku puzzles converge faster than hard ones, correctly detecting accuracy plateaus.
- **Trainability:** Switching from post-norm to pre-norm + residual scaling allows the model to train stably with context lengths and iteration counts where post-norm models fail to utilize signal and pre-norm alone diverges in activation norm.

## Limitations

- **Fixed-point convergence is not guaranteed** for arbitrary learned functions. The oscillation dampening helps, but some inputs may never converge cleanly, requiring a fallback maximum iteration count.
- **Single-level loop** may still be less expressive per parameter than hierarchical approaches (HRM, TRM) on tasks requiring genuinely multi-scale reasoning, even if FPRM currently wins empirically at 7M params.
- The threshold `ε` for halting is a hyperparameter that likely needs tuning per task or difficulty distribution — unclear how to set it robustly across domains.
- Results are primarily on structured puzzle tasks (Sudoku, Maze, ARC-AGI, group theory state-tracking). Generalization to open-ended language reasoning tasks has not been demonstrated.
- Comparisons are within the 7M parameter regime; behavior at scale is unknown.

## Why It Matters

This paper cleanly separates two orthogonal problems in looped reasoning models — *stability* and *adaptivity* — and gives a principled solution to each. The fixed-point halting idea is elegant because it costs nothing at inference: you're already computing `z_{i+1}`, so checking `‖z_{i+1} − z_i‖` is free. The pre-norm + residual scaling insight bridges the literature on deep Transformer training with the looped architecture literature, which had been oddly disconnected.

More broadly, FPRM is part of a wave of work exploring test-time compute scaling *without* Chain-of-Thought verbalization — important because CoT requires curated reasoning traces, special training, and burns through context length. Looped models with principled halting offer a complementary path: latent iterative reasoning that is end-to-end trainable from task supervision alone. If this approach scales, it could sit alongside CoT-based systems as a second paradigm for adaptive inference-time reasoning.

## See Also

[[Universal Transformer]] · [[Adaptive Computation Time]] · [[Chain-of-Thought Prompting]] · [[Deep Equilibrium Models]] · [[Attention Is All You Need]] · [[Test-Time Compute Scaling]] · [[ARC-AGI Benchmark]]
