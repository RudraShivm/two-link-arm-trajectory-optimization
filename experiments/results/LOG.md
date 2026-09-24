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

*Viva point:* "Scaling isn't just dividing variables. The objective has to be scaled consistently too."

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

*(to be filled in)*
