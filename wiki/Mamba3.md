---
created: "2026-05-28"
title: "Mamba-3: Improved Sequence Modeling using State Space Principles"
authors: "Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, Albert Gu"
year: "2026"
arxiv: "2603.15569"
tags: [ssm, state-space-models, mamba, linear-attention, inference-efficiency, language-modeling, recurrent-models, MIMO, complex-valued, discretization]
tldr: "Mamba-3 introduces three principled SSM improvements — exponential-trapezoidal discretization, complex-valued states, and MIMO formulation — achieving +1.8pp over Gated DeltaNet and matching Mamba-2 perplexity at half the state size."
citation_count: 69
aliases: ["Mamba3", "Mamba-3 MIMO", "exponential-trapezoidal SSM"]
---

## TL;DR
Mamba-3 is an inference-first redesign of the Mamba-2 SSM architecture with three principled improvements derived from the continuous-time SSM viewpoint: (1) exponential-trapezoidal discretization for richer dynamics, (2) complex-valued state updates that unlock state-tracking capabilities absent in all prior linear models, and (3) a MIMO generalization that increases hardware utilization 4× during decoding at no latency cost. At 1.5B params, Mamba-3 (MIMO) beats Mamba-2 by +1.9pp and Gated DeltaNet by +1.8pp on downstream accuracy, while matching Mamba-2 perplexity with half its state size.

---

## The Problem
Transformers dominate but are fundamentally bottlenecked:
- **Memory**: KV cache grows linearly with sequence length → expensive long-context inference
- **Compute**: Self-attention is $O(T^2)$ in sequence length $T$

Sub-quadratic alternatives (Mamba-2, Gated DeltaNet, linear attention) have **constant memory** and **linear compute** but suffer three critical gaps:

1. **Quality gap**: Mamba-2 sacrificed expressivity for training simplicity vs. Mamba-1; Mamba-2 heuristic discretization lacked theoretical grounding
2. **Capability gap**: Current linear models fail basic state-tracking (e.g., parity of bit sequences) — they perform at chance on these synthetic tasks
3. **Hardware inefficiency gap**: Decoding is memory-bound with low arithmetic intensity (ratio of FLOPs to memory traffic) — tensor cores sit idle even though the algorithm is theoretically linear

Concretely: Mamba-2 with state size 128 achieves a certain perplexity, but Mamba-3 matches it with state size 64 (half the decode latency). And Mamba-2 scores at random chance on arithmetic state-tracking tasks that Mamba-3 near-perfectly solves.

---

## Core Innovation
**Three orthogonal improvements, all derived from the SSM continuous-time viewpoint (not from the linear attention or test-time regression viewpoint):**

**Analogy**: Think of the original Mamba-2 recurrence as using a Euler step to integrate a differential equation — simple but inaccurate. Mamba-3's exponential-trapezoidal method is like upgrading to a trapezoidal rule: it averages both endpoints of the interval, which implicitly applies a learned convolution over the input. Layered on top, complexifying the hidden state is like giving the system a "rotation" capability — real-valued states can only scale, but complex-valued states can rotate + scale, enabling periodic/cyclic pattern tracking. Finally, MIMO is like upgrading from a one-lane to a multi-lane highway: same road length (state size), but far more throughput.

---

## Architecture / Method

### 2.1 Recap: Mamba-2 Recurrence
The continuous SSM:
$$\dot{h}(t) = A(t)h(t) + B(t)x(t), \quad y(t) = C(t)^\top h(t)$$

Mamba-2 discretizes with scalar $A_t = A_t I$ and heuristic ZOH → Euler:
$$h_t = \alpha_t h_{t-1} + \gamma_t B_t x_t, \quad y_t = C_t^\top h_t$$
where $\alpha_t = e^{\Delta_t A_t} \in (0,1)$ and $\gamma_t = \Delta_t$.

The SSD (State Space Duality) parallel form:
$$Y = (L \odot CB^\top) X$$
where $L$ is a structured causal mask encoding the decay $\alpha_t$ and scale $\gamma_t$.

---

### 3.1 Exponential-Trapezoidal Discretization
**Problem with Euler/ZOH on LTV systems**: The classical ZOH derivation assumed linear-time-invariant (LTI) systems; applying it to time-varying selective SSMs requires an extra heuristic approximation (replacing $\int_{t_k}^{t_{k+1}} e^{A\tau} B d\tau$ with $\Delta_t B_t$).

**Trapezoidal fix**: Instead of approximating the input $B(\tau)x(\tau)$ as constant at the *left* endpoint of the interval, average over *both* endpoints:
$$h_t = \alpha_t h_{t-1} + \gamma_t B_t x_t + \beta_t B_{t-1} x_{t-1}$$
where $\beta_t$ is an additional data-dependent coefficient derived from the trapezoidal rule.

This reveals that the trapezoidal recurrence implicitly applies a **two-tap causal convolution** on the SSM input. More generally, the structured mask becomes:

$$\mathcal{M} = \underbrace{[\text{decay terms}]}_{\text{Mamba-2 part}} \cdot \underbrace{[\text{two-band convolutional mask}]}_{\text{new part}}$$

**Key consequence**: The explicit $B, C$ bias terms in Mamba-3 can **replace the short causal convolution** (conv1d) previously believed essential for recurrent LMs. This simplifies the architecture while adding expressivity.

**Structured mask for Mamba-3**:
- Mamba-2 mask (Eq. 3): product of lower-triangular $\alpha$ decay and diagonal $\gamma$
- Mamba-3 mask: extends this with an additional banded structure from $\beta_t$ terms — still an instance of SSD, hence parallelizable during training

---

### 3.2 Complex-Valued SSM
**Problem**: Real-valued diagonal state-transition $A_t = \alpha_t I$ can only *scale* the state — it cannot *rotate* it. This severely limits state-tracking ability (e.g., counting mod 2 requires representing $\{0, 1\}$ states that a real decay can't distinguish once mixed).

**Solution**: Allow $A_t \in \mathbb{C}$, making the hidden state $h_t \in \mathbb{C}^N$.

The complex update:
$$h_t = (\alpha_t + i\omega_t) h_{t-1} + \gamma_t B_t x_t$$

where $\omega_t \in \mathbb{R}$ is a data-dependent rotation frequency.

**Key insight**: A complex-valued state update is equivalent to a **data-dependent rotary position embedding (RoPE)**:
$$h_t \leftarrow e^{i\theta_t} \cdot |h_t|$$

This means the efficient RoPE computation of Su et al. 2023 applies directly — no new kernels needed for this component.

**Practical efficiency**: The complex arithmetic doubles theoretical FLOPs vs. real arithmetic but the rotation is a lightweight elementwise op. Training and inference overhead is modest.

**Empirical capability**: Mamba-3 with complex states **near-perfectly solves** arithmetic state-tracking tasks (e.g., modular arithmetic, parity). Mamba-2 and Mamba-3 without RoPE score at **random chance** on the same tasks.

---

### 3.3 Multi-Input Multi-Output (MIMO) SSM
**Problem with decoding arithmetic intensity**:
- During decoding, only one new token is processed at a time → batch size 1 in the recurrence
- Mamba-2 state update: $h_t = \alpha_t h_{t-1} + \gamma_t B_t x_t$ is an **outer product** ($B_t \in \mathbb{R}^N$, $x_t \in \mathbb{R}^D$) → $O(ND)$ memory reads, $O(ND)$ FLOPs → low arithmetic intensity → memory bound

**MIMO solution**: Replace SISO (single-input single-output) dynamics with MIMO where $B_t \in \mathbb{R}^{N \times D_{in}}$ and $C_t \in \mathbb{R}^{N \times D_{out}}$ are full matrices rather than vectors:

$$h_t = A_t h_{t-1} + B_t x_t, \quad y_t = C_t^\top h_t$$

with $B_t, C_t$ now dense matrices → state update becomes a **matrix-matrix multiply** → hits tensor cores → dramatically higher arithmetic intensity.

**The key tradeoff**: MIMO increases FLOPs per decode step by up to **4×** at fixed state size $N$, but the increased compute is on matrix multiplies (GPU-efficient) while memory bandwidth stays the same → **wall-clock latency unchanged**, perplexity improves.

Alternatively: MIMO achieves the same quality as SISO with **half the state size** → halves memory bandwidth → lower latency.

---

### 3.4 Architecture Block
The full Mamba-3 block combines:
1. Exponential-trapezoidal recurrence (replaces conv1d + Euler update)
2. Complex-valued hidden state with data-dependent rotation
3. MIMO matrix-based state update (optional variant)
4. Standard Mamba gating and projection structure

**Two variants shipped**:
- **Mamba-3 (SISO)**: improvements 1+2 only
- **Mamba-3 (MIMO)**: all three improvements

Both released with optimized CUDA training + inference kernels at https://github.com/state-spaces/mamba.

---

## Key Results

### Downstream Language Modeling Accuracy (1.5B scale, average over standard LM evals)

| Model | Avg Accuracy | Δ vs Mamba-2 | Δ vs GDN |
|---|---|---|---|
| Transformer | baseline | — | — |
| Mamba-2 | — | 0 | — |
| Gated DeltaNet (GDN) | — | +0.3 | 0 |
| **Mamba-3 (SISO)** | — | +0.9 | **+0.6** |
| **Mamba-3 (MIMO)** | — | **+1.9** | **+1.8** |
| Mamba-3 (MIMO) vs Transformer | — | **+2.2** | — |

### State-Size Efficiency (Perplexity)

| Model | State Size | Perplexity |
|---|---|---|
| Mamba-2 | 128 | X |
| Mamba-3 (MIMO) | 64 | ≈ X (matched) |

→ Same perplexity at **half the state size** = half decode memory bandwidth = lower latency

### State Tracking (Synthetic Tasks)

| Model | Parity/Arithmetic Task |
|---|---|
| Mamba-2 | ~random chance |
| Mamba-3 (no RoPE) | ~random chance |
| **Mamba-3 (with complex/RoPE)** | **near-perfect** |

### Hardware Utilization (Decoding)

| Model | Decode FLOPs vs Mamba-2 | Wall-clock latency change |
|---|---|---|
| Mamba-2 | 1× | baseline |
| Mamba-3 (MIMO) | up to **4×** | ≈ same |

---

## Why It Matters

- **State tracking is now solved for linear models**: The complex-valued state update (equivalent to data-dependent RoPE) gives linear recurrent models a capability that was previously only accessible to Transformers, closing a fundamental expressivity gap demonstrated on parity and modular arithmetic tasks.

- **Inference-first design is the right framing**: Prior SSMs were designed for training throughput; Mamba-3 shows that designing *for decoding arithmetic intensity* (MIMO, reduced state size) yields better quality-latency tradeoffs simultaneously.

- **Kills the conv1d crutch**: The exponential-trapezoidal rule theoretically and empirically shows that the short causal conv1d previously glued into every recurrent LM block was compensating for a poor discretization, not a fundamental necessity.

- **4× more FLOPs at same wall-clock**: MIMO's matrix-multiply-based decoding hits GPU tensor cores instead of memory bandwidth — this is the key to making "theoretically linear" models *actually* hardware-efficient in practice.

- **Half the state size, same quality**: Mamba-3 MIMO at state size 64 matches Mamba-2 at state size 128, which translates directly into lower memory bandwidth pressure and better batching in deployment scenarios.

---

## Connections to Other Work

### Builds directly on:
- [[Transformers Are SSMs|Mamba-2]] — SSD framework, scalar-diagonal state transition, hardware-efficient SSM
- [[Mamba|Mamba-1]] — selective SSM, whose ZOH discretization is now formalized as "exponential-Euler" in Mamba-3's framework
- [[RoPE]] — the complex-valued state update is shown equivalent to data-dependent rotary embeddings, enabling efficient computation

### Competes with:
- [[Gated DeltaNet (Yang et al. 2025)]] — current strongest linear model baseline; Mamba-3 SISO beats it by +0.6pp, MIMO by +1.8pp
- [[Linear attention|Linear Attention]] — simplest SSD instance; Mamba-3 significantly outperforms
- [[HGRN2]], [[RetNet]], [[Gated Linear Attention]] — other sub-quadratic recurrent models on the quality-efficiency frontier

### Enables / appears in:
- [[Jamba|Hybrid SSM-Transformer]] — Mamba-2 layers were incorporated into Kimi Team 2025, NVIDIA 2025 (Hymba), Hunyuan 2025, Qwen 2025 hybrid models; Mamba-3 layers are drop-in improvements
- [[Transformers Are SSMs|State-Space Duality (SSD)]] — Mamba-3's mask is proven to be a valid SSD instance, enabling the same parallel training algorithms

### Addresses limitations diagnosed by:
- Grazzi et al. 2025 — showed Mamba-2 fails state-tracking
- Sarrof et al. 2024 — proved theoretical limits of real-valued SSMs on parity tasks

---

## Limitations

1. **Partial text**: The provided paper text is truncated — Section 3 methodology is cut off mid-sentence and Sections 4-6 (experiments, connections, ablations) are entirely missing. Some specific numbers (absolute perplexity values, full benchmark table) are not available from the provided text.

2. **Complex arithmetic overhead**: While the paper claims the complex RoPE computation is efficient, doubling the state dimension in the complex domain could have non-trivial overhead on hardware that handles complex numbers as two reals — this may vary by implementation quality.

3. **MIMO training complexity**: The MIMO formulation requires full matrix $B_t, C_t$ projections. The increased parameter count and FLOP requirement during
