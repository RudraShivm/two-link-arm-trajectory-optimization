# Trajectory Optimization for a Planar Two-Link Arm

Study guide: problem → theory → experiments → what the numbers mean.  
You do not need to read source code to understand this document.

---

## Problem statement

Given a simulated planar two-link robot arm governed by rigid-body dynamics, compute open-loop joint torques that move the arm from a prescribed start state $x_A$ to a prescribed goal state $x_B$ in a fixed time $T$, while respecting torque and kinematic limits and (when present) obstacle constraints, and while keeping control effort small.

The continuous problem is transcribed into a nonlinear program in three ways—**single shooting (RK4)**, **trapezoidal collocation**, and **Hermite–Simpson collocation**—then solved on identical tasks so that miss distance, physical fidelity under independent ODE replay, and runtime can be compared.

This is a comparative numerical study on a small model inspired by Cardona-Ortiz & Arechavaleta (2025), not a new algorithm and not a physical robot.

---

## How to read this project (no code required)

| Order | What | Why |
|------:|------|-----|
| 1 | Sections 0–2 below (problem + OCP) | What we optimize and why |
| 2 | Section 3 (three methods) + Section 4 (comparison matrix) | How shooting differs from collocation |
| 3 | Section 5 (reality check) | Why “optimizer success” ≠ “physics OK” |
| 4 | [`RESULTS.md`](RESULTS.md) | Main three-method benchmark tables |
| 5 | Latest figures: `output/rest_to_rest_comparison.png`, `output/rest_to_rest_animation.mp4` | Visual of the proposal demo |
| 6 | [`experiments/results/LOG.md`](../experiments/results/LOG.md) and plots in that folder | Follow-on work: faster trapezoidal + accuracy/obstacle fixes |

Skip the Python until you need implementation detail. Theory and experiment logs are enough for viva-level understanding.

---

## 0. What we built and what it means

### Proposal core (CSE 402 comparative study)

1. **Dynamics model.** Two rigid links, gravity, Coriolis coupling: $M(q)\ddot{q}+C(q,\dot{q})\dot{q}+g(q)=\tau$.
2. **Three transcriptions** of the same optimal-control problem (Sections 3.1–3.3).
3. **Same tasks, side by side** (rest-to-rest, swing-up, obstacle, high-speed).
4. **Metrics:** terminal miss (end-effector), open-loop DOP853 drift, solve time, control effort $\int\|\tau\|^2\,dt$.
5. **Artifacts:** comparison plots and a synchronized three-panel animation.

**Headline lesson from the baseline (rest-to-rest, $N=20$).**  
All three methods can hit the target in the NLP. Hermite–Simpson uses the least effort. Trapezoidal is usually the fastest among the collocation methods. Shooting often looks “perfect” at the endpoint but **drifts badly** when the same torques are replayed in an independent high-order ODE integrator—so collocation is more trustworthy for physical open-loop replay on this problem.

### Follow-on work (merged PR: trapezoidal improvements)

Not required by the one-page proposal, but part of the current repo:

| Track | Goal | Achievement (in plain words) |
|-------|------|------------------------------|
| **S1–S5** | Same trapezoidal math, faster NLP | ~10–40× speedup at large $N$; warm start is fastest and most reliable |
| **A1** | Better state shape between nodes | Quadratic interpolant consistent with trapezoidal defects |
| **A2** | Confirm theory | Trapezoidal global error behaves like $O(h^2)$ on fine grids |
| **A3** | Whole-arm clearance | Obstacle can constrain both links, not only the hand |
| **A4** | Midpoint obstacles | Clearance also checked at interval midpoints (less “tunneling”) |

Details and tables: `experiments/results/LOG.md`.

### Fresh vs stale figures

- **Fresh (regenerated with the current solvers):** `output/rest_to_rest_comparison.png`, `output/rest_to_rest_animation.mp4`.
- **Stale (old runs):** `output/obstacle_*` from earlier dates—do not cite as current.
- **From the merged PR (valid, but not the proposal triple-comparison):** `experiments/results/*.png`.

---

## 1. System Dynamics of the Planar Two-Link Robot Arm

### 1.1 Generalized Coordinates and State Representation
Consider a two-link planar serial manipulator operating in a vertical gravitational field ($g_0 = 9.81\,\text{m/s}^2$). The arm consists of two rigid links of lengths $l_1, l_2$, masses $m_1, m_2$, with centers of mass located at distances $r_1, r_2$ from their respective proximal joints, and moments of inertia $I_1, I_2$ about their centers of mass.

The generalized coordinates are the relative joint angles:
$$q = \begin{bmatrix} \theta_1 \\ \theta_2 \end{bmatrix} \in \mathbb{R}^2$$

where $\theta_1$ is the angle of link 1 measured counter-clockwise from the positive horizontal $X$-axis, and $\theta_2$ is the relative angle of link 2 with respect to link 1. The state vector $x(t) \in \mathbb{R}^4$ and control input vector $u(t) \in \mathbb{R}^2$ are defined as:
$$x(t) = \begin{bmatrix} q(t) \\ \dot{q}(t) \end{bmatrix} = \begin{bmatrix} \theta_1(t) \\ \theta_2(t) \\ \dot{\theta}_1(t) \\ \dot{\theta}_2(t) \end{bmatrix}, \quad u(t) = \tau(t) = \begin{bmatrix} \tau_1(t) \\ \tau_2(t) \end{bmatrix}$$

### 1.2 Lagrangian Formulation
The equations of motion are derived via Euler-Lagrange equations:
$$\frac{d}{dt} \left( \frac{\partial \mathcal{L}}{\partial \dot{q}} \right) - \frac{\partial \mathcal{L}}{\partial q} = \tau$$

where the Lagrangian $\mathcal{L} = \mathcal{T} - \mathcal{V}$.

#### Kinetic Energy $\mathcal{T}(q, \dot{q})$
The Cartesian positions of the centers of mass are:
$$\begin{aligned}
p_{c1} &= \begin{bmatrix} r_1 \cos\theta_1 \\ r_1 \sin\theta_1 \end{bmatrix} \\
p_{c2} &= \begin{bmatrix} l_1 \cos\theta_1 + r_2 \cos(\theta_1 + \theta_2) \\ l_1 \sin\theta_1 + r_2 \sin(\theta_1 + \theta_2) \end{bmatrix}
\end{aligned}$$

Differentiating with respect to time yields linear velocities $v_{c1} = \dot{p}_{c1}$ and $v_{c2} = \dot{p}_{c2}$. Total kinetic energy includes translational and rotational components:
$$\mathcal{T} = \frac{1}{2} m_1 \|v_{c1}\|^2 + \frac{1}{2} I_1 \dot{\theta}_1^2 + \frac{1}{2} m_2 \|v_{c2}\|^2 + \frac{1}{2} I_2 (\dot{\theta}_1 + \dot{\theta}_2)^2 = \frac{1}{2} \dot{q}^\top M(q) \dot{q}$$

#### Potential Energy $\mathcal{V}(q)$
Taking the datum at $y = 0$:
$$\mathcal{V}(q) = m_1 g_0 r_1 \sin\theta_1 + m_2 g_0 \left( l_1 \sin\theta_1 + r_2 \sin(\theta_1 + \theta_2) \right)$$

### 1.3 Manipulator Equations of Motion
Carrying out the derivatives yields the standard matrix form:
$$M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q) = \tau$$

1. **Inertia Matrix $M(q) = M(q)^\top > 0$**:
   $$M(q) = \begin{bmatrix} M_{11} & M_{12} \\ M_{21} & M_{22} \end{bmatrix}$$
   $$\begin{aligned}
   M_{11}(q) &= m_1 r_1^2 + I_1 + m_2 (l_1^2 + r_2^2 + 2 l_1 r_2 \cos\theta_2) + I_2 \\
   M_{12}(q) &= M_{21}(q) = m_2 (r_2^2 + l_1 r_2 \cos\theta_2) + I_2 \\
   M_{22}(q) &= m_2 r_2^2 + I_2
   \end{aligned}$$

2. **Coriolis and Centrifugal Vector $C(q, \dot{q})\dot{q}$**:
   Letting $h = m_2 l_1 r_2 \sin\theta_2$:
   $$C(q, \dot{q})\dot{q} = \begin{bmatrix} -h (2 \dot{\theta}_1 \dot{\theta}_2 + \dot{\theta}_2^2) \\ h \dot{\theta}_1^2 \end{bmatrix}$$
   The skew-symmetry property $\dot{M}(q) - 2C(q, \dot{q}) = -(\dot{M}(q) - 2C(q, \dot{q}))^\top$ ensures exact power balance:
   $$\frac{d}{dt} \left( \frac{1}{2}\dot{q}^\top M(q)\dot{q} + \mathcal{V}(q) \right) = \dot{q}^\top \tau$$

3. **Gravity Torque Vector $g(q) = \nabla_q \mathcal{V}(q)$**:
   $$g(q) = \begin{bmatrix} (m_1 r_1 + m_2 l_1) g_0 \cos\theta_1 + m_2 r_2 g_0 \cos(\theta_1 + \theta_2) \\ m_2 r_2 g_0 \cos(\theta_1 + \theta_2) \end{bmatrix}$$
   With the default unit parameters, peak static $|g|$ is about $19.6\,\text{N}\cdot\text{m}$ (horizontal outstretched pose), so under-torqued swing-up with $\tau_{\max}=6$ is intentionally below stall.

4. **Continuous State-Space Representation**:
   $$\dot{x}(t) = f(x(t), u(t)) = \begin{bmatrix} \dot{q}(t) \\ M(q(t))^{-1} \big( u(t) - C(q(t), \dot{q}(t))\dot{q}(t) - g(q(t)) \big) \end{bmatrix}$$

---

## 2. Optimal Control Problem Formulation

The trajectory optimization problem over fixed horizon $t \in [0, T]$ is formulated in the Bolza form:
$$\min_{x(t), u(t)} \quad J = \int_0^T \left( \|u(t)\|^2 + w_{dq} \|\dot{q}(t)\|^2 \right) dt$$
$$\begin{aligned}
\text{subject to} \quad & \dot{x}(t) - f(x(t), u(t)) = 0, && \forall t \in [0, T] \\
& x(0) - x_A = 0, && x(T) - x_B = 0 \\
& -\tau_{\max} \le u(t) \le \tau_{\max}, && \forall t \in [0, T] \\
& q_{\min} \le q(t) \le q_{\max}, && \forall t \in [0, T] \\
& -dq_{\max} \le \dot{q}(t) \le dq_{\max}, && \forall t \in [0, T] \\
& g_{\text{path}}(x(t), u(t)) \ge 0, && \forall t \in [0, T]
\end{aligned}$$

---

## 3. Discretization and Transcription Methods

Direct transcription converts the infinite-dimensional continuous optimal control problem into a finite-dimensional Nonlinear Program (NLP):
$$\min_z \psi(z) \quad \text{s.t.} \quad c_{\text{eq}}(z) = 0, \quad c_{\text{ineq}}(z) \ge 0, \quad z_L \le z \le z_U$$

```
                         TRANSCRIPTION SPECTRUM

   [ Single Shooting ]               [ Trapezoidal Collocation ]          [ Hermite-Simpson Collocation ]
  • Decision: u_0...u_{N-1}          • Decision: x_0,u_0...x_N,u_N        • Decision: x_k, u_k + midpoints
  • ODE simulation in loop           • Piecewise linear dynamics          • Cubic Hermite spline + Simpson
  • O(h^4) RK4 step                  • O(h^2) defect constraints          • O(h^4) defect constraints
  • Dense, sensitive Jacobian        • Sparse block-banded NLP            • Sparse, highly accurate NLP
```

### 3.1 Method 1: Single Direct Shooting (RK4)

* **Decision Vector**: $z = [u_0^\top, u_1^\top, \dots, u_{N-1}^\top]^\top \in \mathbb{R}^{2N}$.
* **Forward Integration**: Given initial condition $x_0 = x_A$, the trajectory is computed sequentially:
  $$x_{k+1} = x_k + \frac{h}{6} (k_1 + 2k_2 + 2k_3 + k_4)$$
  where
  $$\begin{aligned}
  k_1 &= f(x_k, u_k) \\
  k_2 &= f(x_k + \frac{h}{2} k_1, u_k) \\
  k_3 &= f(x_k + \frac{h}{2} k_2, u_k) \\
  k_4 &= f(x_k + h k_3, u_k)
  \end{aligned}$$
* **Terminal Constraint**: Only the endpoint condition is enforced as an NLP constraint:
  $$c_{\text{term}}(z) = x_N(z) - x_B = 0 \in \mathbb{R}^4$$
* **Sensitivity**: Because $x_N$ is a long composition of maps, $\partial x_N/\partial u_0$ can grow or shrink exponentially. On long or oscillatory tasks (e.g. under-torqued swing-up), shooting often fails to converge.

### 3.2 Method 2: Trapezoidal Direct Collocation

Following Cardona-Ortiz & Arechavaleta (2025, Eq. 10-12) and Kelly (2017):
* **Decision Vector**: Both states and controls are optimized simultaneously:
  $$z = [x_0^\top, u_0^\top, x_1^\top, u_1^\top, \dots, x_N^\top, u_N^\top]^\top \in \mathbb{R}^{6(N+1)}$$
* **Defect Constraints**:
  Over each interval $k \in \{0, \dots, N-1\}$ of width $h = T/N$:
  $$\Delta_k = x_{k+1} - x_k - \frac{h}{2} \big( f(x_k, u_k) + f(x_{k+1}, u_{k+1}) \big) = 0 \in \mathbb{R}^4$$
* **Truncation Error**: Local $\mathcal{O}(h^3)$, global $\mathcal{O}(h^2)$.
* **Cost Function** (trapezoidal quadrature of effort; velocity regularization may be added):
  $$\psi(z) = \sum_{k=0}^{N-1} \frac{h}{2} \left( \|u_k\|^2 + \|u_{k+1}\|^2 \right)$$
* **Sparsity**: Each defect depends only on adjacent nodes → block-banded Jacobians.

### 3.3 Method 3: Hermite–Simpson Direct Collocation (Separated Form)

Following Cardona-Ortiz & Arechavaleta (2025, Eq. 14-16) and Betts (2010):
* **Decision Vector**: Mesh nodes and midpoints $\bar{t}_k = t_k + h/2$:
  $$z = [x_0^\top, u_0^\top, \bar{x}_0^\top, \bar{u}_0^\top, \dots, x_N^\top, u_N^\top]^\top \in \mathbb{R}^{6(2N+1)}$$
* **Defect Constraints** (per segment $k$):
  1. Hermite midpoint:
     $$\Delta_k^{\text{mid}} = \bar{x}_k - \frac{1}{2}(x_k + x_{k+1}) - \frac{h}{8} \big( f(x_k, u_k) - f(x_{k+1}, u_{k+1}) \big) = 0$$
  2. Simpson dynamics:
     $$\Delta_k^{\text{dyn}} = x_{k+1} - x_k - \frac{h}{6} \big( f(x_k, u_k) + 4 f(\bar{x}_k, \bar{u}_k) + f(x_{k+1}, u_{k+1}) \big) = 0$$
* **Truncation Error**: Local $\mathcal{O}(h^5)$, global $\mathcal{O}(h^4)$.
* **Cost Function**:
  $$\psi(z) = \sum_{k=0}^{N-1} \frac{h}{6} \left( \|u_k\|^2 + 4 \|\bar{u}_k\|^2 + \|u_{k+1}\|^2 \right)$$

---

## 4. Theoretical Comparison Matrix

| Property | Single Shooting (RK4) | Trapezoidal Collocation | Hermite–Simpson Collocation |
| :--- | :--- | :--- | :--- |
| **Decision Variables** | $2N$ (small) | $6(N+1)$ (moderate) | $12N + 6$ (larger) |
| **NLP Constraint Structure** | Dense terminal (4 equations) | Sparse block-banded ($4N$ eq) | Sparse block-banded ($8N$ eq) |
| **Global Integration Order** | $\mathcal{O}(h^4)$ per step; endpoint map sensitive | $\mathcal{O}(h^2)$ | $\mathcal{O}(h^4)$ |
| **Conditioning of Jacobian** | Can grow like $\mathcal{O}(e^{\lambda T})$ | Polynomial in $1/h$ | Polynomial in $1/h$ |
| **Sensitivity to Initial Guess** | High | Low (linear state guess works) | Low |
| **Nonlinear Path Constraints** | Harder (must sample whole rollout) | Exact at mesh nodes | Exact at nodes and midpoints |
| **Open-Loop Replay** | Exact for its own RK4 stepper; may disagree with adaptive ODE | Truncation drift at coarse $N$ | Higher fidelity at moderate $N$ |

---

## 5. The Physical Reality Check Methodology

Collocation only enforces dynamics at discrete nodes (and midpoints). To measure discretization leakage:

1. Take the optimized controls $u^*(t_k)$.
2. Interpolate $u^*(t)$ continuously (linear in time).
3. Integrate $\dot{x}=f(x,u^*(t))$ from $x_A$ with an independent adaptive solver (`DOP853`, `rtol=atol=10^{-8}`).
4. Compare:
   $$e_{\text{state}}(t) = \| x_{\text{sim}}(t) - x_{\text{opt}}(t) \|, \quad
     e_{\text{cartesian}}(t) = \| p_{ee}(q_{\text{sim}}) - p_{ee}(q_{\text{opt}}) \|$$

This separates “NLP converged” from “the continuous plant follows the plan.”

---

## 6. Experiments at a glance (what each one achieved)

### A. Three-method comparison ([`RESULTS.md`](RESULTS.md))

| Scenario | What we stress | Typical takeaway |
|----------|----------------|------------------|
| **Rest-to-rest** | Easy A→B | All three succeed; HS lowest effort; shooting large DOP853 drift |
| **Obstacle** | Nonlinear path constraint | Collocation handles keep-out naturally; shooting struggles |
| **Swing-up (under-torqued)** | Energy pumping, long horizon | Shooting often fails; collocation finds a swing |
| **High-speed** | Strong Coriolis | Discretization accuracy matters; finer $N$ or higher-order methods help |

**How to read a table row:**  
- *Terminal EE error* ≈ optimizer miss.  
- *Sim max/final drift* ≈ physics fidelity under open-loop replay.  
- *Solve time / iterations* ≈ computational cost.  
- *Effort* ≈ aggressiveness of torques.

### B. Trapezoidal speedups S1–S5 ([`experiments/results/LOG.md`](../experiments/results/LOG.md))

Same defects and cost as the original trapezoidal solver; changes are numerical engineering (bounds, vectorization, sparse Jacobians, scaling, warm start). Result: large speedups with (usually) the same cost; warm start also reduces landing in bad local minima.

### C. Accuracy / correctness A1–A4

- **A1:** Between nodes, state should be a parabola consistent with trapezoidal defects—not a straight line.  
- **A2:** Log–log error vs $h$ confirms ~second-order convergence for trapezoidal collocation.  
- **A3–A4:** Safer obstacle modeling (whole arm + midpoints).

---

## 7. Recommended study path for a viva

1. State the problem (top of this file) in one minute.  
2. Write $M\ddot{q}+C\dot{q}+g=\tau$ and explain $x=[q;\dot{q}]$, $u=\tau$.  
3. Contrast shooting vs collocation (who owns the states?).  
4. Write one trapezoidal defect and one Hermite–Simpson pair.  
5. Explain the reality check and why shooting can “succeed” yet drift.  
6. Quote one rest-to-rest number set (effort + drift + runtime).  
7. Optionally: “we also made trapezoidal ~10–40× faster without changing the math.”

References: Cardona-Ortiz & Arechavaleta (2025); Kelly, SIAM Review (2017); Betts (2010).
