# Two-Link Robot Arm Trajectory Optimization

Compares three numerical methods for planning motion of a planar two-link arm from pose A to pose B under the same time budget and torque limits.

| # | Method | Transcription | Order |
|---|--------|---------------|-------|
| 1 | Single Direct Shooting | Controls only; forward RK4 | O(h⁴) per step |
| 2 | Trapezoidal Direct Collocation | States + controls; trapezoidal defects | O(h²) global |
| 3 | Hermite–Simpson Direct Collocation | States + controls + midpoints | O(h⁴) global |

## Setup

```bash
pip install -r requirements.txt
```

ffmpeg is required only for MP4 export.

## Tests

```bash
PYTHONPATH=. python3 -m unittest discover tests -v
```

## Benchmark

```bash
PYTHONPATH=. python3 run_benchmark.py --scenario rest_to_rest --nodes 30
PYTHONPATH=. python3 run_benchmark.py --scenario all --nodes 30
```

Scenarios: `rest_to_rest`, `swing_up`, `obstacle`, `high_speed`.

## Showcase (plots, animation, web data)

```bash
PYTHONPATH=. python3 generate_showcase.py --scenario rest_to_rest --nodes 30
```

Multi-scenario web catalog:

```bash
PYTHONPATH=. python3 build_full_catalog.py
cd web_viewer && python3 -m http.server 8080
```

Then open http://localhost:8080

## Layout

```
arm_opt/
  dynamics/       # M(q), C(q,dq), g(q), kinematics
  solvers/        # shooting, trapezoidal, hermite–simpson
  scenarios/      # benchmark problems
  analysis/       # metrics + open-loop DOP853 reality check
  visualization/  # plots, MP4, JSON export
tests/
web_viewer/
```

## References

- Cardona-Ortiz, D. and Arechavaleta, G. (2025). Trajectory optimization for highly articulated robots based on sparsity-free local direct collocation. Int. J. Appl. Math. Comput. Sci., 35(4), 577–589.
- Kelly, M. (2017). An Introduction to Trajectory Optimization. SIAM Review, 59(4), 849–904.
- Betts, J.T. (2010). Practical Methods for Optimal Control and Estimation Using Nonlinear Programming. SIAM.

CSE 402 — BUET, Section C-2
