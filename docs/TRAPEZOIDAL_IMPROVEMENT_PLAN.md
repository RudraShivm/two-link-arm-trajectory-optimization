# Trapezoidal Collocation: Improvement Plan

**Team:** Sunanda & Arpa
**Owned file:** [`arm_opt/solvers/trapezoidal.py`](../arm_opt/solvers/trapezoidal.py)
**Goal:** make the trapezoidal solver faster and more correct, prove its accuracy experimentally, and have clear before/after numbers for the report and viva.

---

## Table of contents

1. [Quick recap: what the solver does today](#1-quick-recap-what-the-solver-does-today)
2. [Baseline numbers (before any change)](#2-baseline-numbers-before-any-change)
3. [The full list of improvements](#3-the-full-list-of-improvements)
4. [Who does what](#4-who-does-what)
5. [Ground rules (read before writing code)](#5-ground-rules-read-before-writing-code)
6. [The shared interface (agree on this on day 1)](#6-the-shared-interface-agree-on-this-on-day-1)
7. [Sunanda's tasks: Speed](#7-sunandas-tasks-speed)
8. [Arpa's tasks: Accuracy & Correctness](#8-arpas-tasks-accuracy--correctness)
9. [Stretch goal: Mesh refinement](#9-stretch-goal-mesh-refinement)
10. [Testing checklist](#10-testing-checklist)
11. [Order of work / phases](#11-order-of-work--phases)
12. [What goes in the report and slides](#12-what-goes-in-the-report-and-slides)

---

## 1. Quick recap: what the solver does today

- Time `[0, T]` is chopped into **N intervals**, giving **N+1 points** ("nodes"). With N = 30, h = T/30.
- At every node the unknowns are **4 states** `[θ1, θ2, ω1, ω2]` and **2 torques** `[τ1, τ2]`, so there are 6 per node and `(N+1)·6 = 186` in total, all packed into one flat vector `z`.
- **Minimize** effort `∫(τ1² + τ2² + 0.001·(ω1² + ω2²)) dt`, computed with the composite trapezoidal rule.
- **Subject to** these rules:
  - Start state = A (4 equations), end state = B (4 equations).
  - **Defect** on every interval: `x_{k+1} − x_k − (h/2)(f_k + f_{k+1}) = 0`, so 30 × 4 = 120 equations (line 100).
  - Min/max bounds on every variable.
  - Optional obstacle inequalities (≥ 0) at every node.
- scipy's `SLSQP` optimizer solves it. **No derivatives are supplied**, so scipy estimates them by nudging all 186 variables one by one in every iteration. That is the main reason the solver is slow.

---

## 2. Baseline numbers (before any change)

Measured on the `rest_to_rest` scenario with the current code:

| N | Solve time | Iterations | Final drift (reality check) |
|---|---|---|---|
| 10 | 2.46 s | 46 | 0.9966 m |
| 20 | 6.81 s | 44 | 0.0712 m |
| 40 | 21.92 s | 46 | 0.0094 m |

What these numbers tell us:
- **Iterations stay around 45, but time grows about 3× every time N doubles.** Each iteration gets more expensive, which is exactly what the Speed tasks target.
- **Drift drops quickly as N grows**, which is what the Accuracy tasks will study properly.

> **First thing both of you do:** re-run the baseline on your own laptop (script in [§7, Task S0](#task-s0-together-baseline-script)). Timing depends on the machine, so all "before vs after" numbers must come from the **same** laptop.

---

## 3. The full list of improvements

> **Status:** S1–S5 are implemented in [`arm_opt/solvers/improved_trapezoidal.py`](../arm_opt/solvers/improved_trapezoidal.py)
> (a separate file; the original `trapezoidal.py` is untouched). Results are in
> [`experiments/results/LOG.md`](../experiments/results/LOG.md). Implementation differences from §7:
> the batch physics lives inside the new file (so `manipulator.py` wasn't touched), and S4 also scales the objective.

| # | Improvement | Owner | Difficulty | Status |
|---|---|---|---|---|
| S1 | Fix start/end states using bounds instead of equations | Sunanda | Easy | ✅ Done |
| S2 | Vectorize the physics (all nodes in one call) | Sunanda | Easy | ✅ Done |
| S3 | Sparse Jacobian (analytic objective gradient + colored finite differences) | Sunanda | Medium | ✅ Done ⭐ |
| S4 | Variable scaling | Sunanda | Easy | ✅ Done (objective must be scaled too, see LOG.md) |
| S5 | Warm start (solve coarse first, then refine) | Sunanda | Easy | ✅ Done |
| A1 | Correct (quadratic) state interpolation between nodes | Arpa | Easy | Keep |
| A2 | Convergence-order study (prove O(h²)) | Arpa | Medium | Keep ⭐ |
| A3 | Whole-arm obstacle check (not just the hand) | Arpa | Easy | Keep, **but the obstacle must move first** |
| A4 | Check obstacle at interval midpoints too | Arpa | Medium | Keep |
| X1 | Mesh refinement (adaptive grid) | Whoever finishes first | Hard | Stretch |
| ✗ | Switch to IPOPT / CasADi | — | Hard | **Dropped** |

**Why IPOPT/CasADi is dropped:**
1. It replaces S2 and S3 (CasADi computes sparse derivatives automatically), so that work would be wasted.
2. It makes the benchmark unfair: trapezoidal would use a much stronger optimizer than shooting and Hermite–Simpson, so "trapezoidal is faster" would really mean "IPOPT is faster".
3. It requires learning a new library and rewriting the file.

---

## 4. Who does what

### 👤 Sunanda: "Speed" (make it fast without changing the answer)
S1 → S2 → S3 → S4 → S5

**Final deliverable:** a plot and table of **solve time vs N (10, 20, 40, 80)**, old vs new, plus proof that the **cost is the same** (same answer, faster).

### 👤 Arpa: "Accuracy & Correctness" (prove it works, fix real flaws)
A1 → A2 → A3 → A4

**Final deliverables:**
- A **log-log convergence plot** with a fitted slope of about 2.
- **Before/after pictures** of the arm passing through the obstacle vs going around it.

> The split is by theme, so each of you owns a clear story in the presentation. Swap roles if you prefer, but keep the split by theme.

### How the two halves depend on each other
- **Arpa needs Sunanda's speed** for N = 80 and 160 in the convergence study. Until S3 is merged, Arpa uses N ≤ 40 (about 20 s per solve).
- **Sunanda's Jacobian code (S3) must also cover Arpa's new constraints (A3, A4).** The shared interface in [§6](#6-the-shared-interface-agree-on-this-on-day-1) makes that automatic.

---

## 5. Ground rules (read before writing code)

### 5.1 Keep the old behavior switchable
Every improvement goes behind a flag that is **off by default**:

```python
solver = TrapezoidalCollocationSolver(
    problem,
    fix_endpoints_in_bounds=False,  # S1
    vectorized=False,               # S2
    sparse_jacobian=False,          # S3
    scaling=False,                  # S4
    check_midpoints=False,          # A4
)
```

Why:
- **Before/after numbers need the "before" version to still exist.** Without it, you can't prove your contribution.
- `run_benchmark.py` and the other two solvers keep working exactly as before.
- Once everything is tested, the whole team can decide to turn the flags on by default.

### 5.2 Don't silently change shared files
These files are used by **all three solvers** (i.e. by your other teammates):

| File | Rule |
|---|---|
| `arm_opt/dynamics/manipulator.py` | Only **add** new functions (S2). Never change existing ones. |
| `arm_opt/analysis/reality_check.py` | Only add an **optional** parameter whose default keeps the old behavior (A1). |
| `arm_opt/scenarios/obstacle.py` | Changing the obstacle affects everyone. **Tell the whole team first** (A3). |
| `arm_opt/solvers/base.py` | Avoid changing it. If you must, tell the team. |

### 5.3 Git workflow
```bash
# one-time: each person gets their own branch
git checkout main && git pull
git checkout -b sunanda/speed        # or: arpa/accuracy

# daily
git pull origin main                 # pick up merged work from the other person
# ...work...
git add <files> && git commit -m "S2: add vectorized state_derivative_batch"
git push -u origin sunanda/speed
```
- **One commit per task** (S1, S2, …), each with its task ID in the message. The history then shows who did what.
- **Both of you edit `trapezoidal.py`**, so merge small and often. Merge S3 into `main` as soon as it works, because A4 builds on top of it.
- Arpa should put new helper code in a **new file** (`arm_opt/solvers/trapezoidal_utils.py`) where possible, to reduce merge conflicts.

### 5.4 Record every number immediately
Keep a running log at `experiments/results/LOG.md`. After every task, paste its table there. That log becomes the report.

### 5.5 Timing rules
- Run each timing **3 times and report the median**.
- Close heavy apps (browser, IDE indexing) while timing.
- Always compare on the same laptop.

---

## 6. The shared interface (agree on this on day 1)

Right now constraints are built as plain scipy dicts inside `solve()`. We change that to a list of **constraint blocks**. Each block knows **which nodes each of its rows depends on**:

```python
# inside solve()
blocks = []   # each block is a dict:
# {
#   "type": "eq" or "ineq",
#   "fun": callable z -> np.ndarray of shape (m,),
#   "row_nodes": list of length m; row_nodes[r] = tuple of node indices row r depends on
# }
```

Examples:

| Block | Rows | `row_nodes` |
|---|---|---|
| Start state `x_0 − A` | 4 | `[(0,)] * 4` |
| End state `x_N − B` | 4 | `[(N,)] * 4` |
| Defects | 4N | defect k's 4 rows → `(k, k+1)` |
| Obstacle at nodes | N+1 | row k → `(k,)` |
| Obstacle at midpoints (A4) | N | row k → `(k, k+1)` |

At the end of `solve()`, blocks are converted into scipy constraints:
- If `sparse_jacobian=False`: `{"type": ..., "fun": ...}`, exactly like today.
- If `sparse_jacobian=True`: `{"type": ..., "fun": ..., "jac": <fast sparse jacobian>}`, built by Sunanda's helper from `row_nodes`.

**Arpa's rule:** every new constraint you add must be a block with a correct `row_nodes`. Listing *extra* nodes is safe (just slightly slower). Listing *too few* nodes gives **wrong derivatives**, so if in doubt, include more.

**Sunanda's rule:** do the refactor to blocks **first** (as part of S1) and push it quickly, so Arpa builds on it.

---

## 7. Sunanda's tasks: Speed

### Task S0 (together): Baseline script

Create `experiments/trapezoidal_baseline.py`:

```python
"""Baseline timing of the trapezoidal solver. Run: PYTHONPATH=. python3 experiments/trapezoidal_baseline.py"""
import statistics
from arm_opt.scenarios import SCENARIOS
from arm_opt.solvers.trapezoidal import TrapezoidalCollocationSolver
from arm_opt.analysis.reality_check import perform_reality_check

SCENARIO = "rest_to_rest"
for N in [10, 20, 40]:
    times = []
    for _ in range(3):
        prob = SCENARIOS[SCENARIO]().create_problem(n_nodes=N)
        res = TrapezoidalCollocationSolver(prob).solve()
        times.append(res.solve_time)
    rc = perform_reality_check(prob, res)
    print(f"N={N:3d} success={res.success} median_time={statistics.median(times):.2f}s "
          f"iters={res.iterations} cost={res.cost:.4f} "
          f"viol={res.max_constraint_violation:.1e} final_drift={rc.terminal_cartesian_drift:.4f}")
```

Later, extend the same script with the new flags so it prints old vs new side by side.

---

### Task S1: Fix start/end states via bounds  *(easy, warm-up)*

**Problem.** Lines 91–92 make "x_0 = A" and "x_N = B" into 8 equality constraints. The optimizer must *work* to satisfy them, even though we already know their exact values.

**Fix.** Set the min and max bound of those 8 variables to the same value, so they are locked from the start.

**Steps:**
1. First do the **block refactor** from [§6](#6-the-shared-interface-agree-on-this-on-day-1) with the flag off. Check that results are identical to baseline, then push.
2. Add the `fix_endpoints_in_bounds` flag to `__init__`.
3. In the bounds loop, for node 0 replace the 4 state bounds with `(p.x0[i], p.x0[i])`, and for node N with `(p.xf[i], p.xf[i])`.
4. When the flag is on, **don't add** the start/end blocks.
5. The initial guess already starts at A and ends at B (a straight line), so nothing else changes.

**Pitfall.** scipy (version ≥ 1.9, you have 1.18) removes locked variables internally before calling SLSQP. Test that it still works together with S3's `jac`. If you get a shape error, tell Arpa, and temporarily test S3 with this flag off.

**Measure.** Iterations and time, flag on vs off, N = 20 and 40.
**Honest expectation.** The gain is small. It's a warm-up, not the headline.

---

### Task S2: Vectorize the physics  *(easy)*

**Problem.** Lines 96–97 loop in Python over all 31 nodes, calling `state_derivative` once per node. Python loops are slow; numpy on whole arrays is fast.

**Fix.** **Add** (don't replace) a batch version to `TwoLinkArm` in `manipulator.py`:

```python
def state_derivative_batch(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Vectorized f(x, u) for many nodes at once. X: (K, 4), U: (K, 2) -> (K, 4)."""
    p = self.p
    q1, q2, dq1, dq2 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    c2, s2 = np.cos(q2), np.sin(q2)

    # Mass matrix entries (same formulas as mass_matrix)
    m11 = p.m1 * p.r1**2 + p.I1 + p.m2 * (p.l1**2 + p.r2**2 + 2.0 * p.l1 * p.r2 * c2) + p.I2
    m12 = p.m2 * (p.r2**2 + p.l1 * p.r2 * c2) + p.I2
    m22 = p.m2 * p.r2**2 + p.I2

    # Coriolis (same as coriolis_vector)
    h = p.m2 * p.l1 * p.r2 * s2
    cor1 = -h * (2.0 * dq1 * dq2 + dq2**2)
    cor2 = h * dq1**2

    # Gravity (same as gravity_vector)
    c1, c12 = np.cos(q1), np.cos(q1 + q2)
    g1 = (p.m1 * p.r1 + p.m2 * p.l1) * p.g * c1 + p.m2 * p.r2 * p.g * c12
    g2 = p.m2 * p.r2 * p.g * c12

    # Solve the 2x2 system M * ddq = rhs with the explicit inverse formula
    b1 = U[:, 0] - cor1 - g1
    b2 = U[:, 1] - cor2 - g2
    det = m11 * m22 - m12**2
    ddq1 = (m22 * b1 - m12 * b2) / det
    ddq2 = (-m12 * b1 + m11 * b2) / det
    return np.column_stack([dq1, dq2, ddq1, ddq2])
```

Then in `trapezoidal.py`, when `vectorized=True`:

```python
f_vals = self.arm.state_derivative_batch(states, controls)          # (N+1, 4)
defects = states[1:] - states[:-1] - 0.5 * h * (f_vals[:-1] + f_vals[1:])   # (N, 4), no loop
```

**Test (required):** for 1000 random states and torques, `state_derivative_batch` must match the old `state_derivative` loop (`np.allclose(..., rtol=1e-10, atol=1e-10)`; expect differences around 1e-13). See [§10](#10-testing-checklist).

**Measure.** Time per call of `equality_constraints` (use `timeit`), old vs new, plus total solve time.

---

### Task S3: Sparse Jacobian  ⭐ *(medium, the main contribution)*

**Problem.** Without derivatives, SLSQP nudges each of the 186 variables separately in every iteration. That's 186 full constraint evaluations per iteration.

**Key insight.** Defect k only depends on nodes k and k+1. So the Jacobian (the table of "how each rule changes when each variable changes") is almost all zeros, with a thin diagonal band:

```
            node0  node1  node2  node3 ...
defect 0  [  ███    ███     ·      ·   ]
defect 1  [   ·     ███    ███     ·   ]
defect 2  [   ·      ·     ███    ███  ]
```

This has two parts.

#### Part (a): Exact gradient of the objective
The cost is `cost = (h/2) Σ w_k L_k` with weights `w = [1, 2, 2, ..., 2, 1]` and `L_k = τ1² + τ2² + c(ω1² + ω2²)`. Differentiating gives `∂cost/∂τ_k = h·w_k·τ_k` and `∂cost/∂ω_k = h·w_k·c·ω_k`, and the angles don't appear, so their gradient is 0.

```python
def objective_grad(z):
    states, controls = self.unpack(z)
    h = self.dt
    w = np.full(self.N + 1, 2.0)
    w[0] = w[-1] = 1.0
    G = np.zeros((self.N + 1, self.node_vars))
    G[:, 2:4] = h * w[:, None] * p.effort_weight_dq * states[:, 2:4]
    G[:, 4:6] = h * w[:, None] * controls
    return G.flatten()

# then: minimize(fun=objective, jac=objective_grad, ...)
```

#### Part (b): Constraint Jacobian with "colored" finite differences
Think of the classroom analogy: nudge **all odd nodes at once**, then **all even nodes at once**. Two nodes that are not neighbors never appear in the same defect, so their effects don't mix. With 6 variables per node × 2 groups, that's **12 evaluations instead of 186**.

Put these generic helpers in `arm_opt/solvers/sparse_fd.py`. They work for **any** block with `row_nodes`, including Arpa's new ones:

```python
import numpy as np

def build_pattern(row_nodes, n_nodes, node_vars=6):
    """Boolean (m, n) matrix: S[r, c] = True if row r may depend on variable c."""
    S = np.zeros((len(row_nodes), n_nodes * node_vars), dtype=bool)
    for r, nodes in enumerate(row_nodes):
        for k in nodes:
            S[r, k * node_vars:(k + 1) * node_vars] = True
    return S

def color_columns(S):
    """Group columns so no two columns in a group share a row (greedy)."""
    groups = []  # each entry: [list_of_columns, rows_already_used]
    for col in range(S.shape[1]):
        rows = S[:, col]
        for g in groups:
            if not np.any(g[1] & rows):
                g[0].append(col)
                g[1] |= rows
                break
        else:
            groups.append([[col], rows.copy()])
    return [np.array(g[0]) for g in groups]

def sparse_fd_jacobian(fun, z, S, groups, rel_step=1e-7):
    """Forward-difference Jacobian using one evaluation per column group."""
    f0 = fun(z)
    J = np.zeros((f0.size, z.size))
    steps = rel_step * np.maximum(1.0, np.abs(z))
    for cols in groups:
        z_pert = z.copy()
        z_pert[cols] += steps[cols]
        df = fun(z_pert) - f0
        for c in cols:
            rows = S[:, c]
            J[rows, c] = df[rows] / steps[c]
    return J
```

Usage inside `solve()`: build `S` and `groups` **once** per block (before calling `minimize`), then:

```python
{"type": blk["type"], "fun": blk["fun"],
 "jac": lambda z, blk=blk, S=S, G=G: sparse_fd_jacobian(blk["fun"], z, S, G)}
```
(The `blk=blk, S=S, G=G` default arguments "freeze" the current values inside the loop. It's the same Python trick as `make_path_con` in the original code.)

Print `len(groups)` once. For the defect block it should be **12**.

**Tests (required):**
1. For a random `z`, `sparse_fd_jacobian` must match scipy's dense finite difference (`scipy.optimize.approx_fprime` row by row, or `scipy.optimize._numdiff.approx_derivative`) to about `1e-5`.
2. `objective_grad` must match a dense finite difference of `objective` to about `1e-5`.
3. A full solve with `sparse_jacobian=True` must give the **same cost** as baseline (relative difference < 1e-3) and still succeed.

**Measure.** Solve time and iterations for N = 10, 20, 40, 80, sparse vs baseline. N = 80 baseline may take minutes; run it once.

**Honest expectation.** A large speedup, but less than 15×, because SLSQP also spends time on its own internal (dense) linear algebra. Report whatever you actually measure.

---

### Task S4: Variable scaling  *(easy)*

**Problem.** Angles are around 1, velocities around 10, torques around 30. Optimizers behave best when all variables have similar size (around 1).

**Fix.** Let the optimizer work with scaled variables `y = z / s`, where `s` holds a "typical size" for each variable:

```python
s = np.tile([1.0, 1.0, p.dq_max, p.dq_max, p.tau_max, p.tau_max], self.N + 1)

fun_y  = lambda y: objective(y * s)
grad_y = lambda y: objective_grad(y * s) * s              # chain rule
con_y  = lambda y: blk_fun(y * s)
jac_y  = lambda y: blk_jac(y * s) * s[None, :]            # scale each column
bounds_y = [(lo / si, hi / si) for (lo, hi), si in zip(bounds, s)]
y0 = z0 / s
# after solving: z_final = res.x * s
```

**Pitfalls:**
- Every function passed to `minimize` must use the scaled version, including bounds and the initial guess.
- Remember to unscale `res.x` before `unpack`.

**Measure.** Iterations and success rate, on and off, especially on the hard scenarios (`swing_up`, `high_speed`).

**Honest expectation.** It might help a lot, a little, or not at all. **Report it either way.** "We tested scaling and it did not help, because…" is still a valid result.

---

### Task S5: Warm start  *(easy)*

**Problem.** The initial guess is a straight line with **zero velocity everywhere**, which is physically impossible (the angle changes but the speed is 0). The optimizer has to travel far from it.

**Fix.** Solve on a coarse grid first, stretch that answer onto the fine grid, and use it as the guess.

**Step 1.** Let `solve()` accept a guess:

```python
def solve(self, max_iter=400, ftol=1e-6, initial_guess=None):
    ...
    if initial_guess is not None:
        t_old, X_old, U_old = initial_guess
        t_new = self.problem.time_grid
        init_states = np.column_stack([np.interp(t_new, t_old, X_old[:, i]) for i in range(4)])
        init_controls = np.column_stack([np.interp(t_new, t_old, U_old[:, i]) for i in range(2)])
    else:
        ...existing straight-line + gravity guess...
```

**Step 2.** Add a helper (in `trapezoidal.py` or `trapezoidal_utils.py`):

```python
import dataclasses

def solve_with_warm_start(problem, levels=(10,), **solver_kwargs):
    """Solve on coarse grids first, each one warm-starting the next, ending at problem.n_nodes."""
    guess, total_time, total_iters = None, 0.0, 0
    for N in list(levels) + [problem.n_nodes]:
        sub = dataclasses.replace(problem, n_nodes=N)
        res = TrapezoidalCollocationSolver(sub, **solver_kwargs).solve(initial_guess=guess)
        total_time += res.solve_time
        total_iters += res.iterations
        guess = (res.time, res.state, res.control)
    res.info["final_level_iterations"] = res.iterations
    res.solve_time = total_time           # fair: includes the coarse solves
    res.iterations = total_iters
    return res
```

**Pitfall (fairness).** The reported time **must include the coarse solves**. Otherwise the comparison is cheating.

**Measure:**
- Cold vs warm start at N = 40 and 80: total time, total iterations, and final-level iterations.
- Does warm start make hard scenarios (`swing_up`, `high_speed`, `under_torqued`) converge where cold start fails? Run each scenario both ways and make a success/fail table.

---

### Sunanda's final deliverable
1. `experiments/trapezoidal_speed.py`, which produces:
   - a **table** (N, baseline time, each flag's time, all flags on, cost for each)
   - a **plot** `experiments/results/speed_vs_N.png` (x = N, y = solve time, log scale on y, one line per configuration)
2. Unit tests from [§10](#10-testing-checklist).
3. A short paragraph per task in `LOG.md`: what changed, the numbers, and why it helped (or didn't).

---

## 8. Arpa's tasks: Accuracy & Correctness

### Task A1: Correct (quadratic) interpolation  *(easy)*

**Problem.** The solver only gives values **at nodes**. To know where the arm is *between* nodes, the reality check currently draws straight lines ([`reality_check.py` line 77](../arm_opt/analysis/reality_check.py#L77)). But the trapezoidal method assumes the **velocity** changes linearly between nodes, which means the **state** follows a **parabola**, not a straight line (Kelly 2017, §4).

**Formula.** For a time `t = t_k + τ` with `0 ≤ τ ≤ h`:

```
x(t) = x_k + f_k · τ + (τ² / (2h)) · (f_{k+1} − f_k)
u(t) = u_k + (τ / h) · (u_{k+1} − u_k)          (torque stays linear)
```

Sanity check: at `τ = h` this gives `x_k + (h/2)(f_k + f_{k+1})`, which is exactly `x_{k+1}` when the defect is 0. ✅

**Code** (new file `arm_opt/solvers/trapezoidal_utils.py`):

```python
import numpy as np

def trapezoidal_interpolate(arm, t_nodes, states, controls, t_query):
    """State (quadratic) and control (linear) between nodes, as implied by trapezoidal collocation."""
    t_query = np.atleast_1d(t_query)
    f = np.array([arm.state_derivative(x, u) for x, u in zip(states, controls)])
    k = np.clip(np.searchsorted(t_nodes, t_query, side="right") - 1, 0, len(t_nodes) - 2)
    h = (t_nodes[k + 1] - t_nodes[k])[:, None]
    tau = (t_query - t_nodes[k])[:, None]
    x = states[k] + f[k] * tau + (tau**2 / (2.0 * h)) * (f[k + 1] - f[k])
    u = controls[k] + (tau / h) * (controls[k + 1] - controls[k])
    return x, u
```
(It works with uneven spacing too, which the stretch goal needs.)

**Hook into the reality check** without changing the default. Add an optional parameter to `perform_reality_check`:

```python
def perform_reality_check(problem, result, num_eval_points=200, state_interp_fn=None):
    ...
    if state_interp_fn is None:
        state_interpolator = interp1d(t_opt, x_opt, axis=0, kind="linear")   # old behavior
        x_opt_interp = state_interpolator(t_sim)
    else:
        x_opt_interp = state_interp_fn(t_sim)
```

**Be precise about what this changes** (the viva may ask):
- **Max drift** changes, because it compares the simulation against the interpolated plan.
- **Final drift does NOT change**, because it compares the simulation's end with the target and never uses state interpolation.
- The torque interpolation (line 51) is linear, which is **already correct for trapezoidal**. It's actually unfair to shooting (which holds torque constant within each step) and Hermite–Simpson (which uses curved torques), but that is the other teams' business. Mention it to them.

**Measure.** Max drift, linear vs quadratic, N = 10, 20, 40. Plot one joint angle over time showing the nodes, the straight-line version, the parabola version, and the high-accuracy simulation.

---

### Task A2: Convergence-order study  ⭐ *(medium, the main contribution)*

**Theory.** The trapezoidal rule has error **O(h²)**: halve h (double N) and the error drops about **4×**. On a log-log plot of error vs h, the points should lie on a line with **slope ≈ 2**.

**Script:** `experiments/trapezoidal_convergence.py`

```python
import numpy as np
import matplotlib.pyplot as plt
from arm_opt.scenarios import SCENARIOS
from arm_opt.solvers.trapezoidal import TrapezoidalCollocationSolver
from arm_opt.analysis.reality_check import perform_reality_check

Ns = [10, 15, 20, 30, 40]            # add 80, 160 once Sunanda's S3 is merged
hs, errs = [], []
for N in Ns:
    prob = SCENARIOS["rest_to_rest"]().create_problem(n_nodes=N)
    res = TrapezoidalCollocationSolver(prob).solve(max_iter=1000, ftol=1e-10)
    rc = perform_reality_check(prob, res)
    hs.append(prob.dt)
    errs.append(rc.terminal_cartesian_drift)
    print(N, res.success, res.max_constraint_violation, rc.terminal_cartesian_drift)

slope, _ = np.polyfit(np.log(hs), np.log(errs), 1)
print(f"fitted order = {slope:.2f}")
# plot: plt.loglog(hs, errs, "o-"); also draw a reference line with slope 2
```

**Error measures to report:**
1. **Final drift** (simulation end vs target). Uses linear torques, which is consistent with trapezoidal, so it's fair as-is.
2. **Max state error** between the simulation and the **quadratic interpolant** from A1.

**Pitfalls (important):**
- **Optimizer tolerance must be much tighter than the error you're measuring.** At large N the discretization error becomes tiny. If `ftol` is loose, the solver stops early and the curve **flattens**, and you'd measure the optimizer instead of the method. Use `ftol=1e-10`, and check that `max_constraint_violation` is far below the drift.
- The reality check itself uses `rtol=atol=1e-8`, so errors below about 1e-7 can't be trusted.
- **Don't expect exactly 4× per doubling.** Baseline drift went 0.997 → 0.071 → 0.0094 (14×, then 7.6×). Very coarse grids (N = 10) aren't in the "asymptotic" regime yet, and the optimal answer itself shifts a bit with N. That's why you **fit a line through many N values** instead of comparing pairs. If the fitted slope isn't close to 2, investigate and explain (e.g. drop N = 10, check solver tolerance) instead of hiding it.
- Run it on at least **two scenarios** (`rest_to_rest` and `high_speed`).

**Optional, but great for the report:** add Hermite–Simpson to the same plot (expected slope ≈ 4). This needs the HS team's help: for a fair slope, HS must be replayed with its own curved (quadratic) torques. With linear torques its slope gets stuck around 2.

---

### Task A3: Whole-arm obstacle check  *(easy, but coordinate first)*

**Problem.** The obstacle constraint only checks the **hand** (end-effector). The links can pass right through the obstacle:

```
      ╭───╮
  ■═══│═══│═══●      hand (●) is outside ✅, link (═) goes through ❌
      ╰───╯
```

**⚠️ Catch: the current obstacle makes a whole-arm check impossible.**
The arm moves from θ1 = −45° to +45°, so the shoulder **must** pass θ1 = 0°. At that moment link 1 lies on the x-axis from (0, 0) to (1, 0). The obstacle is centered at (1.2, 0) with radius 0.355 (including the margin), so link 1 comes **within 0.2 m of the center, inside the circle**. With a whole-arm check **no solution exists**, and every solver will fail.

**Steps:**
1. **Demonstrate the bug first** (this is part of the contribution). Take the current trapezoidal solution, compute the minimum distance from each link to the obstacle at every time, and plot the frame where a link is inside.
2. **Agree with the whole team** on a new obstacle (the scenario is shared). A candidate that looks feasible: **center (1.6, 0), radius 0.3**. Link 1 (reach 1.0) can never touch it, and link 2 can dodge by bending the elbow. **Verify** by solving before committing to it.
3. Add a flag to `ObstacleScenario`, e.g. `check_full_body=False`. When it's on, add one constraint function **per sample point** on **both** links:

```python
def make_point_constraint(link, s):
    """link = 1: point = s * elbow;  link = 2: point = elbow + s * (ee - elbow)."""
    def con(x, u):
        p_elbow, p_ee = forward_kinematics(x[:2], params)
        point = s * p_elbow if link == 1 else p_elbow + s * (p_ee - p_elbow)
        return float(np.sum((point - self.obs_center) ** 2) - (self.obs_radius + margin) ** 2)
    return con

path_constraints = [make_point_constraint(link, s)
                    for link in (1, 2) for s in (0.25, 0.5, 0.75, 1.0)]
```
Each of these is a per-node constraint, so its `row_nodes` is `(k,)` for row k.

**Pitfall: gaps between sample points.** A circle can slip between two sample points. With samples 0.25 m apart and radius 0.3, it can dig in by up to about **0.03 m**. Fixes: use a margin ≥ 0.03, use more sample points, or use the exact point-to-segment distance (project the center onto the segment and clamp to `[0, 1]`).

**Measure / show.** Before/after animation frames (the arm through the obstacle vs around it), minimum clearance over time for both links, and solve time (more constraints means slower).

---

### Task A4: Check the obstacle between nodes (midpoints)  *(medium)*

**Problem.** Constraints are checked only **at the nodes**. Between two nodes the hand can cut the corner:

```
   ●           ●        nodes are outside ✅
     ╲ ╭───╮ ╱
      ╲│   │╱           but the path between them cuts through ❌
       ╰───╯
```

**Fix.** Use A1's parabola to compute the state at the **midpoint** of every interval, and require the constraint there too. At `τ = h/2` the formula simplifies to:

```
x_mid = x_k + (h/8) · (3·f_k + f_{k+1})
u_mid = (u_k + u_{k+1}) / 2
```

**Code** (inside `solve()`, when `check_midpoints=True`, for each path constraint `fn`):

```python
def midpoint_con(z, fn=fn):
    states, controls = self.unpack(z)
    f = self.arm.state_derivative_batch(states, controls)       # from S2
    x_mid = states[:-1] + (self.dt / 8.0) * (3.0 * f[:-1] + f[1:])
    u_mid = 0.5 * (controls[:-1] + controls[1:])
    return np.array([fn(x_mid[k], u_mid[k]) for k in range(self.N)])

blocks.append({"type": "ineq", "fun": midpoint_con,
               "row_nodes": [(k, k + 1) for k in range(self.N)]})
```
Because `row_nodes` is set correctly, Sunanda's sparse Jacobian covers it automatically.

**How to prove it matters:**
1. Write a checker: sample the parabola at **20 points per interval** and compute the minimum clearance over the whole trajectory.
2. Corner cutting is most visible with **few nodes**, so try N = 8, 10, 15 on the obstacle scenario.
3. Table: N, minimum clearance with midpoints off vs on (negative = collision).
4. If the default scenario never cuts corners, make the demo case harder (bigger obstacle or smaller N) and **say so honestly** in the report.

---

### Arpa's final deliverable
1. `experiments/trapezoidal_convergence.py` → `experiments/results/convergence.png` plus the fitted slope.
2. `experiments/obstacle_checks.py` → before/after images and clearance tables for A3 and A4.
3. Unit tests from [§10](#10-testing-checklist).
4. A short paragraph per task in `LOG.md`.

---

## 9. Stretch goal: Mesh refinement

Only start this after all your main tasks are done. It fits Arpa best, since it reuses A1 and A4.

**Idea.** Uniform spacing wastes nodes where the motion is gentle. Put more nodes where the error is big.

**Algorithm:**
1. Solve on a coarse grid.
2. For each interval, estimate its error: evaluate the parabola's velocity `dx/dt` at the midpoint and compare it with `f(x_mid, u_mid)`. The size of the mismatch is the interval's error.
3. Split every interval whose error is above a threshold into two.
4. Re-solve, warm-started from the previous solution (S5).
5. Repeat until all intervals are below the threshold, or a node budget is reached.

**Code changes needed:** the solver assumes one fixed `h`. It must switch to an array `h_k = t_{k+1} − t_k`, used in the defects, the objective weights, and A4's midpoint formula. The result's `time` must be the non-uniform grid (the reality check already uses `result.time`, so it keeps working).

**Show.** "Same accuracy with X adaptive nodes as with Y uniform nodes", plus a plot of where the nodes cluster (likely where the arm accelerates or brakes hard).

---

## 10. Testing checklist

Create `tests/test_trapezoidal_improvements.py`. Run everything with:
```bash
PYTHONPATH=. python3 -m unittest discover tests
```

| Test | Owner | Pass condition |
|---|---|---|
| Existing tests still pass | both | unchanged |
| All flags off gives the same result as the original code | both | cost identical |
| `state_derivative_batch` matches the loop on 1000 random inputs | Sunanda | `np.allclose(..., rtol=1e-10, atol=1e-10)` |
| `objective_grad` matches dense finite difference | Sunanda | < 1e-5 |
| `sparse_fd_jacobian` matches dense finite difference (defects + obstacle blocks) | Sunanda | < 1e-5 |
| Defect block uses 12 color groups | Sunanda | `len(groups) == 12` |
| Each speed flag gives the same cost as baseline | Sunanda | relative diff < 1e-3, success = True |
| Warm-start time includes coarse solves | Sunanda | `solve_time` ≥ sum of level times |
| `trapezoidal_interpolate` passes through every node | Arpa | error ≤ max defect |
| Midpoint formula equals the general formula at τ = h/2 | Arpa | < 1e-12 |
| Full-body obstacle scenario is feasible and the solution has clearance ≥ 0 everywhere | Arpa | min clearance ≥ −1e-6 |
| Midpoint check: solution has clearance ≥ 0 at the midpoints | Arpa | min midpoint clearance ≥ −1e-6 |

Use the small test problem from `tests/test_solvers.py` (N = 15) for most tests so they run fast.

---

## 11. Order of work / phases

| Phase | Sunanda | Arpa | Together |
|---|---|---|---|
| **0: Setup** | | | Read this doc. Run S0 baseline on both laptops. Create branches. Agree on the block interface (§6). |
| **1** | Block refactor (flags off, identical results) → push. S1. | A1 interpolation + reality-check hook. | Sunanda merges the refactor to `main` early. |
| **2** | S2 vectorize. S3 sparse Jacobian. | A2 convergence study with N ≤ 40. Demo the A3 bug. Propose the new obstacle to the whole team. | Merge S3 → Arpa pulls it and extends A2 to N = 80, 160. |
| **3** | S4 scaling. S5 warm start. | A3 full-body check. A4 midpoints. | Cross-review each other's code (each explains their part to the other; it's viva practice). |
| **4: Wrap-up** | Speed plot and table. | Convergence plot and obstacle images. | Final `LOG.md`, report section, slides. Stretch goal if there is time. |

**Cross-review rule:** before merging, the *other* person reads the diff and must be able to explain it. In the viva, either of you may be asked about any part.

---

## 12. What goes in the report and slides

**Story (about 5 slides):**
1. **Recap:** trapezoidal collocation in one picture (nodes, defects, trapezoid).
2. **Speed (Sunanda):** "Each defect only couples neighbors, so the Jacobian is banded. Coloring cut derivative evaluations from 186 to 12 per iteration." Show the speed-vs-N plot and the cost-is-unchanged table, plus the scaling and warm-start results.
3. **Accuracy (Arpa):** "We verified O(h²) experimentally." Show the log-log plot with the fitted slope, and explain any deviation from exactly 2.
4. **Correctness (Arpa):** "The original obstacle check let the arm pass through the obstacle, and even corner-cut between nodes." Show before/after images, and mention that the obstacle had to move because the original one made a whole-arm check impossible.
5. **Lessons & limitations:** what didn't help (be honest), why IPOPT was deliberately not used (fair comparison), and the benchmark's torque-interpolation bias toward trapezoidal (reported to the other teams).

**Every claim needs a number from `LOG.md`.** If a task didn't improve anything, say so and explain why. That still counts as a result.
