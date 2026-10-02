# Results Log: Trapezoidal Improvements

Written findings to go with the auto-generated tables in
[trapezoidal_comparison.md](trapezoidal_comparison.md).
Code: [`arm_opt/solvers/improved_trapezoidal.py`](../../arm_opt/solvers/improved_trapezoidal.py).
Reproduce: `PYTHONPATH=. python3 experiments/compare_trapezoidal.py`.

---

## Sunanda: S1–S5 (Speed)

### Headline numbers

**rest_to_rest, solve time vs N** (same cost in every row, to 4+ digits):

| N | Original | Improved (S1–S4) | Improved + warm start (S1–S5) |
|---|---|---|---|
| 10 | 2.55 s | 0.18 s (14×) | 0.18 s (14×) |
| 20 | 7.01 s | 0.46 s (15×) | 0.33 s (21×) |
| 40 | 21.30 s | 2.15 s (10×) | 0.98 s (22×) |
| 80 | 119.87 s | 9.90 s (12×) | **2.95 s (41×)** |

**All scenarios at N = 30:**

| Scenario | Original | Improved + warm start | Speedup | Same answer? |
|---|---|---|---|---|
| rest_to_rest | 14.53 s | 0.61 s | 24× | ✅ identical cost |
| obstacle | 28.78 s | 3.74 s | 7.7× | different but valid path, **29% cheaper** (239.2 vs 335.3) |
| high_speed | 34.63 s | 0.81 s | 43× | ✅ identical cost |
| under_torqued | 12.73 s | 0.42 s | 30× | ✅ identical cost |
| reversal | 93.54 s | 1.66 s | 56× | ✅ identical cost |
| swing_up | 19.23 s | 0.83 s | 23× | ✅ identical cost |

### Each improvement's contribution (rest_to_rest, N = 40, applied one after another)

| Step | Time | Total speedup | What happened |
|---|---|---|---|
| Original | 21.54 s | 1.0× | |
| + S1 endpoints as bounds | 20.32 s | 1.1× | Small gain, as expected (8 fewer equations out of 168). |
| + S2 vectorized physics | 7.05 s | 3.1× | Biggest single jump. Python loops were the bottleneck. |
| + S3 sparse Jacobian | 3.44 s | 6.3× | 12 evaluations per Jacobian instead of 246. |
| + S4 scaling | 2.12 s | 10.2× | Same iterations, but less time per iteration. |
| + S5 warm start | 0.98 s | 22.0× | More iterations in total, but most run on cheap coarse grids. |

### Surprise 1: scaling only the variables made things WORSE

The plan said to divide velocities by 15 and torques by 30. On its own, that made
rest_to_rest (N = 40) go from **46 to 115 iterations**, and on high_speed it even
converged to a **worse answer** (cost 1414 instead of 1210).

**Why:** SLSQP starts by assuming the objective curves by about the same amount (about 1) in every
direction. After dividing torque by 30, the effort curves **900× more sharply** in the
scaled-torque direction, so SLSQP's starting assumption is badly wrong and it
wastes iterations learning the real shape.

**Fix:** also divide the objective by `h · τ_max²`, which brings that curvature back to about 2–4.
We tested four choices of objective scale. `h · τ_max²` was the only one that never
made things worse:

| Scenario, N | No scaling | Variables only | Variables + objective ÷ h·τ² |
|---|---|---|---|
| rest_to_rest 40 | 46 it | 115 it | 46 it |
| high_speed 40 | 111 it | 103 it (worse answer) | **56 it** |
| swing_up 20 | 94 it | 56 it | **48 it** |

*Takeaway:* "Scaling isn't just dividing variables. The objective has to be scaled consistently too."

### Surprise 2: some scenarios have more than one answer

**reversal:** Improved (S1–S4) *without* warm start finds cost **1019.8**, while the original
finds **566.3**. We checked that 1019.8 is a genuine local minimum ("a different valley"),
not a bug: restarting the original-style solver from that answer accepts it after 1 iteration.
Even S1 vs S1+S2, which compute the same numbers up to rounding at the 13th decimal, can land
in different valleys. So this problem is very sensitive to tiny numerical differences.

**Warm start fixes it.** With S5 we get 572.5 / 566.3 / 564.2 at N = 20 / 30 / 40, matching the
original. The coarse solve finds the right valley, and the fine solve just refines it.

**obstacle:** all three solvers return **valid** paths (minimum clearance ≥ 0, so the hand never
enters the circle), but they bend the elbow in different directions and have different costs.
The improved + warm-start version found the cheapest one (239 vs 335). This is normal for
problems with obstacles, which usually have several locally-best paths.

**Takeaway:** use `solve_with_warm_start()` as the default. It's the fastest *and* the most
reliable configuration we tested.

### Caveats (be honest in the report)
- Timings are from one laptop; median of 3 runs (the original at N = 80 was timed once).
- Warm-start times **include** the coarse solves, so the comparison is fair.
- The speedup at a fixed N varies by scenario (7.7× to 56×). The obstacle scenario gains least,
  because its obstacle constraint is still evaluated node by node in Python (not vectorized).

### Tests
`tests/test_improved_trapezoidal.py` has 14 tests, all passing:
- the batch physics matches the loop
- the exact gradient matches a numerical one
- the sparse Jacobian matches a dense one (including obstacle rows)
- the defect Jacobian uses exactly 12 colour groups
- every flag alone, and all together, gives the original's answer
- warm start counts all its time and iterations

---

## Arpa: A1–A4 (Accuracy & Correctness)

### A1: Quadratic (trapezoidal-consistent) state interpolation

**What changed.**
- New file [`arm_opt/solvers/trapezoidal_utils.py`](../../arm_opt/solvers/trapezoidal_utils.py) adds
  `trapezoidal_interpolate()`. Trapezoidal collocation assumes f = ẋ changes *linearly* inside an
  interval, so the state follows the parabola
  `x(t) = x_k + f_k·τ + (τ²/2h)(f_{k+1} − f_k)`. The torque stays linear.
- [`reality_check.py`](../../arm_opt/analysis/reality_check.py) gets an optional
  `state_interp_fn=None` parameter. The default keeps the old straight-line behaviour, so the
  other solvers are unaffected.

**Numbers** (`rest_to_rest`, warm start, `ftol=1e-10`;
reproduce with `PYTHONPATH=. python3 experiments/trapezoidal_interpolation.py`):

Reality check (one open-loop simulation over the whole motion):

| N | max state drift (linear) | max state drift (quadratic) | max EE drift (linear) | max EE drift (quadratic) | final drift (both) |
|---|---|---|---|---|---|
| 10 | 7.3376 | 6.7070 | 0.9967 m | 0.9967 m | 0.9967 m |
| 20 | 3.1085 | 3.1085 | 0.1197 m | 0.1202 m | 0.0712 m |
| 40 | 0.7969 | 0.6870 | 0.0318 m | 0.0318 m | 0.0094 m |

Local interpolation error (simulation restarted from the node at every interval, so accumulated
drift is removed and only the interpolant's shape is measured):

| N | linear | quadratic | linear / quadratic |
|---|---|---|---|
| 10 | 3.7295 | 2.5842 | 1.4× |
| 20 | 1.5034 | 1.3468 | 1.1× |
| 40 | 0.6851 | 0.2558 | 2.7× |

Plot: [`interpolation.png`](interpolation.png) (θ1 and ω1 at N = 10: nodes, linear, quadratic, simulation).

**Findings.**
- **Final drift is identical** with both interpolations, as expected. It compares the
  simulation's end with the target and never uses the state interpolant.
- **Max drift barely changes** (at most 1.2× better). Over the whole motion, the open-loop
  simulation drifts away from the plan, and that accumulated drift is much larger than the
  interpolation's shape error. So max drift mostly measures the drift, not the interpolation.
- **Locally, the parabola is always better**, and the gap grows as N increases (2.7× at N = 40).
  In the plot, the linear version cuts straight across the ω1 peaks; the parabola follows
  the curve.
- Conclusion: use the quadratic interpolant whenever the trapezoidal state is needed *between*
  nodes. A2 (convergence) and A4 (midpoint obstacle check) build on it.

**Tests** ([`tests/test_trapezoidal_improvements.py`](../../tests/test_trapezoidal_improvements.py), 7 tests, all pass):
interpolant passes through every node (error ≤ max defect), the end of each parabola lands on the
next node, it is continuous across nodes, its slope at each node equals f(x_k, u_k), the default
reality check is unchanged, and the hook changes max drift but not final drift.

### A2: Convergence-order study

**What changed.**
- New script [`experiments/trapezoidal_convergence.py`](../trapezoidal_convergence.py). For
  N = 10, 15, 20, 30, 40, 80, 160 on `rest_to_rest` and `high_speed`, it solves with warm start
  (`ftol=1e-10`, `max_iter=1000`), runs the reality check, and fits the slope of log(error) vs
  log(h). Results are cached in `convergence.json`, and `--replot` redraws without re-solving.
- `fit_order()` added to [`trapezoidal_utils.py`](../../arm_opt/solvers/trapezoidal_utils.py).
- Error measures: **final drift** (simulation end vs target) and **max state error** (simulation
  vs the A1 quadratic interpolant, over the whole trajectory).

**Numbers** (reproduce: `PYTHONPATH=. python3 experiments/trapezoidal_convergence.py`, about 5 min;
N = 160 alone takes 100–150 s):

| scenario | N | h [s] | expected ratio (O(h²)) | final drift [m] | ratio | max state error | ratio | max violation |
|---|---|---|---|---|---|---|---|---|
| rest_to_rest | 10 | 0.10000 |  | 9.967e-01 |  | 6.718e+00 |  | 4.4e-14 |
| rest_to_rest | 15 | 0.06667 | 2.25× | 2.304e-01 | 4.3× | 4.662e+00 | 1.4× | 1.2e-14 |
| rest_to_rest | 20 | 0.05000 | 1.78× | 7.120e-02 | 3.2× | 3.108e+00 | 1.5× | 4.5e-14 |
| rest_to_rest | 30 | 0.03333 | 2.25× | 2.028e-02 | 3.5× | 1.281e+00 | 2.4× | 2.5e-13 |
| rest_to_rest | 40 | 0.02500 | 1.78× | 9.441e-03 | 2.1× | 6.870e-01 | 1.9× | 2.7e-13 |
| rest_to_rest | 80 | 0.01250 | 4.00× | 1.681e-03 | 5.6× | 1.718e-01 | 4.0× | 2.3e-13 |
| rest_to_rest | 160 | 0.00625 | 4.00× | 3.927e-04 | 4.3× | 4.354e-02 | 3.9× | 8.7e-13 |
| high_speed | 10 | 0.04500 |  | 1.287e-01 |  | 1.760e+00 |  | 1.1e-13 |
| high_speed | 15 | 0.03000 | 2.25× | 4.160e-02 | 3.1× | 2.196e+00 | 0.8× | 4.6e-14 |
| high_speed | 20 | 0.02250 | 1.78× | 1.465e-02 | 2.8× | 1.239e+00 | 1.8× | 5.9e-14 |
| high_speed | 30 | 0.01500 | 2.25× | 2.174e-03 | 6.7× | 5.390e-01 | 2.3× | 9.4e-14 |
| high_speed | 40 | 0.01125 | 1.78× | 1.604e-03 | 1.4× | 3.171e-01 | 1.7× | 1.4e-13 |
| high_speed | 80 | 0.00562 | 4.00× | 4.022e-04 | 4.0× | 7.962e-02 | 4.0× | 4.7e-13 |
| high_speed | 160 | 0.00281 | 4.00× | 1.136e-04 | 3.5× | 2.032e-02 | 3.9× | 8.9e-14 |

(ratio = error at the previous N / error at this N. O(h²) predicts (h_prev/h)², i.e. 4× per doubling of N.)


| scenario | metric | fitted order (all N) | fitted order (N ≥ 20) |
|---|---|---|---|
| rest_to_rest | final_drift | 2.81 | 2.48 |
| rest_to_rest | max_state_error | 1.90 | 2.04 |
| high_speed | final_drift | 2.56 | 2.17 |
| high_speed | max_state_error | 1.76 | 1.97 |

Plot: [`convergence.png`](convergence.png) (log-log, both scenarios, with a slope-2 reference line).

**Findings.**
- **Trapezoidal collocation is second order, confirmed.** Max state error, fitted over N ≥ 20,
  has order **2.04** (`rest_to_rest`) and **1.97** (`high_speed`). From N = 40 → 80 → 160 the error
  drops by 4.0× and 3.9× per doubling in both scenarios, almost exactly the 4× the theory predicts.
- **The optimizer is not distorting the result.** Max constraint violation is ≤ 9e-13 at every N,
  at least 8 orders of magnitude below the smallest error measured (1.1e-4 m). The smallest error
  is also far above the reality check's own floor (about 1e-7).
- **Coarse grids (N ≤ 15) are not yet in the asymptotic regime.** There the ratios are irregular
  (for example high_speed 10 → 15 gives 0.8×, meaning the error *grew*). This is why the headline
  fit uses N ≥ 20. With all N the fit gives 1.90 / 1.76.
- **Final drift is noisier** (fitted 2.48 / 2.17). It measures a single point, the arm's end
  position, and errors from different parts of the motion can partly cancel there, so it jumps
  around (high_speed 30 → 40 is only 1.4×). It still settles to about 4× per doubling at N ≥ 80.
  Max state error is the more reliable measure of order, because it takes the worst point over the
  whole trajectory.
- The trapezoidal optimum itself also changes with N (cost 243.1 at N = 10 → 219.1 at N = 160 for
  `rest_to_rest`), which is another reason ratios between single pairs of N are not exactly 4×.

**Tests** (2 new, all pass): `fit_order` recovers slopes 2 and 4 on exact data, and on the small test
problem (N = 20, 40, 80) both error measures have a fitted order between 1.7 and 2.5, with the
constraint violation < 1e-3 × the error.

### A3: Whole-arm obstacle check

**What changed.**
- [`arm_opt/scenarios/obstacle.py`](../../arm_opt/scenarios/obstacle.py): new flag
  `check_full_body=False`. When it is on, the hand-only constraint is replaced by one constraint
  per link: the **exact** squared distance from the obstacle center to the link segment (the
  projection clamped to [0, 1]), minus (r + 5 mm)². Link 2 ends at the hand, so it also covers the
  hand. The exact distance has no gaps, unlike sample points along the link (the plan's sample
  points 0.25 apart could let the circle dig in by about 0.03 m). It adds 2 rows per node, and each
  row uses only node k, so `row_nodes = (k,)` and the sparse Jacobian covers it automatically.
- **Defaults are unchanged** (obstacle (1.2, 0), r = 0.35, hand-only), so the benchmark and the
  other solvers behave exactly as before. The new obstacle is available as
  `ObstacleScenario.FULL_BODY_CENTER` / `FULL_BODY_RADIUS` and only used when asked for.
- [`trapezoidal_utils.py`](../../arm_opt/solvers/trapezoidal_utils.py): `link_clearances()`
  measures the clearance of link 1, link 2 and the hand (negative = collision).
- Script: [`experiments/obstacle_checks.py`](../obstacle_checks.py). Clearance is measured on the
  A1 quadratic interpolant at 3000 time samples, so it also sees between-node motion.

**Part 1: the bug in the original scenario** (N = 30, warm start):
[`obstacle_bug_original.png`](obstacle_bug_original.png)
- The solver succeeds (cost 239.2) and the hand stays outside **at the nodes** (+0.005 m).
- But **both links pass through the obstacle**: minimum clearance −0.150 m for link 1 and link 2,
  with a link inside the obstacle at 5 of the 31 nodes.
- Between nodes even the hand goes 0.015 m inside (this is corner cutting, which A4 addresses).
- With the whole-arm check on this obstacle the problem is **infeasible** (success = False, max
  violation 19.1). As predicted in the plan: the arm must pass θ1 = 0, where link 1 lies on the
  x-axis only 0.2 m from the obstacle's center.

**Choosing the new obstacle.** The plan's candidate (1.6, 0), r = 0.3 turned out to be useless for
the demo: the minimum-effort arm never goes near it (link 2 clearance +0.216 m even with the hand-only
check), so hand-only and whole-arm give the same answer. A search over centers and radii
(conditions: link 1 can never reach it, the hand-only solution shows the bug, and the whole-arm solve
works for every N and start) picked **(1.5, −0.3), r = 0.3**. Link 1 can never touch it
(1.53 − 0.3 = 1.23 m > l1 = 1 m). (1.35, 0.25) also showed the bug, but its whole-arm solve failed at
N = 20 and N = 40, so it was rejected.

**Part 2: before vs after on the new obstacle**:
[`obstacle_full_body.png`](obstacle_full_body.png) (cold start, N = 30)

| N | start | check | success | cost | min link clearance (dense) | solve time |
|---|---|---|---|---|---|---|
| 20 | cold | hand only | True | 417.2 | −0.300 m ❌ | 8.3 s |
| 20 | cold | whole arm | True | 172.5 | +0.230 m | 2.2 s |
| 20 | warm | hand only | True | 172.5 | +0.230 m | 2.0 s |
| 20 | warm | whole arm | True | 172.5 | +0.230 m | 9.0 s |
| 30 | cold | hand only | True | 414.8 | −0.300 m ❌ | 15.2 s |
| 30 | cold | whole arm | True | 171.4 | +0.230 m | 3.1 s |
| 30 | warm | hand only | True | 171.4 | +0.230 m | 4.8 s |
| 30 | warm | whole arm | True | 171.4 | +0.230 m | 3.2 s |
| 40 | cold | hand only | True | 394.7 | −0.299 m ❌ | 13.3 s |
| 40 | cold | whole arm | True | 171.0 | +0.230 m | 6.8 s |
| 40 | warm | hand only | True | 171.0 | +0.230 m | 3.2 s |
| 40 | warm | whole arm | True | 171.0 | +0.230 m | 10.8 s |

**Findings.**
- **With the hand-only check, "success = True" does not mean collision-free.** From a cold start
  (the way the original benchmark runs the solver) all 3 hand-only runs swing the hand around the
  outside of the obstacle while **link 2 cuts through its center** (−0.30 m = the full radius).
- **The whole-arm check gives a collision-free path every time** (6/6: all N, cold and warm), with
  +0.23 m clearance. The arm folds its elbow and passes inside the obstacle.
- The hand-only problem has **two local minima**: the colliding "swing around" path (cost about 400) and
  the folded path (cost about 171). Warm start happens to find the folded one, which avoids the obstacle
  by luck, not because anything checks the links. The whole-arm check removes the colliding
  minimum, so the result no longer depends on the starting guess.
- Solve time: the whole-arm check adds 2 constraint rows per node instead of 1. It is *faster* than
  hand-only for cold start (the colliding minimum was expensive to reach) and 1–3× slower for warm start.
- The whole-arm constraint is still only enforced **at nodes**. A4 checks the midpoints too.

**Tests** (4 new, all pass): segment-distance geometry (projection and clamping), `link_clearances`
matches forward kinematics on 200 random poses, the default scenario is unchanged (1 constraint, same
value), and the whole-arm solution on the new obstacle (N = 20) succeeds with node clearance ≥ −1e-6.

**⚠ Tell the team:** the shared `ObstacleScenario` gained a flag but its defaults are untouched.
If the team wants the whole-arm check in the main benchmark, all three solvers must switch to the new
obstacle together, because the old one is infeasible with it.

### A4: Obstacle check at interval midpoints

**What changed.**
- [`improved_trapezoidal.py`](../../arm_opt/solvers/improved_trapezoidal.py): new flag
  `check_midpoints=False` and a `midpoints(z)` method. With the flag on, every path constraint gets a
  second block evaluated at the middle of every interval, using the A1 parabola at τ = h/2:
  `x_mid = x_k + (h/8)(3 f_k + f_{k+1})`, `u_mid = (u_k + u_{k+1})/2`. Row k depends on nodes k and
  k+1, so `row_nodes = (k, k+1)` and the S3 sparse Jacobian covers it automatically (12 colour
  groups, same as the defects). It works with `solve_with_warm_start(..., check_midpoints=True)` and
  with the A3 whole-arm constraints. Default off, so all earlier results are unchanged.
- [`trapezoidal_utils.py`](../../arm_opt/solvers/trapezoidal_utils.py): `path_clearance()` samples
  the parabola at 20 points per interval (the checker the plan asks for).
- Script: `PYTHONPATH=. python3 experiments/obstacle_checks.py --part a4`.

**Numbers** (original obstacle scenario, hand-only check, warm start, `ftol=1e-8`):

| N | midpoints | success | cost | min hand clearance at nodes | min hand clearance between nodes | solve time |
|---|---|---|---|---|---|---|
| 8 | off | True | 244.5 | +0.0050 m | −0.3018 m ❌ | 0.9 s |
| 8 | on | True | 253.5 | +0.0050 m | −0.3331 m ❌ | 1.5 s |
| 10 | off | True | 317.0 | +0.0050 m | −0.2613 m ❌ | 1.4 s |
| 10 | on | True | 322.7 | +0.0050 m | −0.1563 m ❌ | 2.1 s |
| 15 | off | True | 232.8 | +0.0050 m | −0.2301 m ❌ | 2.7 s |
| 15 | on | True | 300.7 | +0.0050 m | −0.0131 m ❌ | 3.1 s |
| 20 | off | True | 306.9 | +0.0050 m | −0.1258 m ❌ | 2.1 s |
| 20 | on | True | 304.2 | +0.0050 m | −0.0028 m ❌ | 6.4 s |
| 30 | off | True | 239.2 | +0.0050 m | −0.0150 m ❌ | 2.7 s |
| 30 | on | True | 297.2 | +0.0050 m | **+0.0021 m** ✅ | 4.1 s |

Plot: [`obstacle_midpoints.png`](obstacle_midpoints.png) (N = 15: hand path near the obstacle, off vs on,
and clearance over time).

**Findings.**
- **Corner cutting is real and large.** At every N the hand is exactly on the safety margin at the
  nodes (+0.005 m), so "all constraints satisfied" is true, yet between two nodes the hand goes up to
  **0.30 m** inside an obstacle of radius 0.35. At N = 15 the hand simply jumps from one side of the
  obstacle to the other in a single interval (see the plot).
- **Checking midpoints fixes most of it for N ≥ 15.** The worst penetration drops from −0.230 → −0.013 m
  (N = 15), −0.126 → −0.003 m (N = 20), and −0.015 m → **+0.002 m, collision-free** (N = 30).
- **It does not help at N = 8 (and only partly at N = 10).** With so few nodes, one interval is long
  enough that the path can pass *between* a node and a midpoint. Checking only the midpoint halves the
  spacing between checked points; it does not guarantee the whole curve. More check points per
  interval (e.g. quarter points), or simply more nodes, would be needed.
- **Being safe costs effort.** With midpoints on, the cost is usually higher (N = 30: 239.2 → 297.2,
  +24%) because the cheap solution was only cheap because it cut through the obstacle. Small
  exceptions (N = 20: 306.9 → 304.2) come from SLSQP landing in different local minima; the cost also
  jumps around with N for the same reason (the hand-only problem has several local minima, see A3).
- On the A3 whole-arm scenario (new obstacle) midpoints change nothing, because that path already has
  +0.23 m clearance everywhere, so there is no corner to cut.
- Solve time grows 1.1–3× (N more constraint rows per path constraint).

**Tests** (5 new, all pass): the midpoint formula equals the general A1 formula at τ = h/2 (< 1e-12),
the flag is off by default, the midpoint block's sparse Jacobian matches a dense finite difference and
uses 12 colour groups, a midpoint solve (N = 15) has clearance ≥ −1e-6 at both nodes and midpoints, and
`path_clearance` samples include the nodes.
