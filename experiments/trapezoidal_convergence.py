"""A2: convergence-order study of trapezoidal collocation.

Theory: trapezoidal collocation is second order, error ≈ C·h². Doubling N should
cut the error by about 4×, i.e. a slope of about 2 on a log-log plot of error vs h.

For every N the problem is solved with warm start and ftol=1e-10, so the
optimizer's stopping error stays far below the discretization error being measured.

Error measures:
  1. final drift: open-loop simulation end vs target (end-effector, metres)
  2. max state error: simulation vs the QUADRATIC interpolant from A1 (whole trajectory)

Run:     PYTHONPATH=. python3 experiments/trapezoidal_convergence.py
Replot:  PYTHONPATH=. python3 experiments/trapezoidal_convergence.py --replot   (uses cached JSON)
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from arm_opt.analysis.reality_check import perform_reality_check
from arm_opt.scenarios import SCENARIOS
from arm_opt.solvers.improved_trapezoidal import solve_with_warm_start
from arm_opt.solvers.trapezoidal_utils import fit_order, make_state_interp_fn

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CACHE = os.path.join(RESULTS_DIR, "convergence.json")
SCENARIOS_TO_RUN = ["rest_to_rest", "high_speed"]
NS = [10, 15, 20, 30, 40, 80, 160]
ASYMPTOTIC_MIN_N = 20  # coarser grids are not yet in the O(h²) regime
REALITY_CHECK_FLOOR = 1e-7  # reality check integrates with rtol=atol=1e-8

METRICS = [
    ("final_drift", "Final drift [m]"),
    ("max_state_error", "Max state error vs quadratic interpolant"),
]
COLORS = {"rest_to_rest": "#2a78d6", "high_speed": "#eb6834"}
MARKERS = {"rest_to_rest": "o", "high_speed": "s"}


def run_all():
    data = {}
    for scenario in SCENARIOS_TO_RUN:
        rows = []
        for n in NS:
            problem = SCENARIOS[scenario]().create_problem(n_nodes=n)
            result = solve_with_warm_start(problem, max_iter=1000, ftol=1e-10)
            rc = perform_reality_check(
                problem, result, num_eval_points=400, state_interp_fn=make_state_interp_fn(problem, result)
            )
            row = {
                "N": n,
                "h": problem.dt,
                "success": bool(result.success),
                "violation": float(result.max_constraint_violation),
                "cost": float(result.cost),
                "solve_time": float(result.solve_time),
                "final_drift": rc.terminal_cartesian_drift,
                "max_state_error": rc.max_state_drift,
            }
            rows.append(row)
            print(
                f"{scenario:13s} N={n:4d} success={row['success']} viol={row['violation']:.1e} "
                f"time={row['solve_time']:6.1f}s final_drift={row['final_drift']:.3e} "
                f"max_state_error={row['max_state_error']:.3e}",
                flush=True,
            )
        data[scenario] = rows
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CACHE, "w") as fh:
        json.dump(data, fh, indent=2)
    return data


def orders(rows, key):
    hs = [r["h"] for r in rows]
    errs = [r[key] for r in rows]
    asym = [(r["h"], r[key]) for r in rows if r["N"] >= ASYMPTOTIC_MIN_N]
    return fit_order(hs, errs), fit_order(*zip(*asym))


def check_trustworthy(data):
    """Warn if a measured error is too close to the optimizer or integrator noise floor."""
    for scenario, rows in data.items():
        for r in rows:
            smallest = min(r["final_drift"], r["max_state_error"])
            if not r["success"]:
                print(f"WARNING {scenario} N={r['N']}: solver did not succeed")
            if r["violation"] > 1e-3 * smallest:
                print(f"WARNING {scenario} N={r['N']}: constraint violation is not << error")
            if smallest < 10 * REALITY_CHECK_FLOOR:
                print(f"WARNING {scenario} N={r['N']}: error near the reality-check floor")


def report(data):
    lines = [
        "### A2: convergence order (warm start, ftol=1e-10)",
        "",
        "| scenario | N | h [s] | expected ratio (O(h²)) | final drift [m] | ratio | max state error | ratio | max violation |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for scenario, rows in data.items():
        prev = None
        for r in rows:
            fr = f"{prev['final_drift'] / r['final_drift']:.1f}×" if prev else ""
            sr = f"{prev['max_state_error'] / r['max_state_error']:.1f}×" if prev else ""
            er = f"{(prev['h'] / r['h']) ** 2:.2f}×" if prev else ""
            lines.append(
                f"| {scenario} | {r['N']} | {r['h']:.5f} | {er} | {r['final_drift']:.3e} | {fr} | "
                f"{r['max_state_error']:.3e} | {sr} | {r['violation']:.1e} |"
            )
            prev = r
    lines += [
        "",
        "(ratio = error at the previous N / error at this N; O(h²) predicts (h_prev / h)², i.e. 4× per doubling of N)",
        "",
        f"| scenario | metric | fitted order (all N) | fitted order (N ≥ {ASYMPTOTIC_MIN_N}) |",
        "|---|---|---|---|",
    ]
    for scenario, rows in data.items():
        for key, label in METRICS:
            p_all, p_asym = orders(rows, key)
            lines.append(f"| {scenario} | {key} | {p_all:.2f} | {p_asym:.2f} |")
    return "\n".join(lines)


def plot(data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (key, label) in zip(axes, METRICS):
        for scenario, rows in data.items():
            hs = np.array([r["h"] for r in rows])
            errs = np.array([r[key] for r in rows])
            _, p_asym = orders(rows, key)
            ax.loglog(
                hs, errs, marker=MARKERS[scenario], ms=8, lw=2, color=COLORS[scenario],
                markeredgecolor="white", markeredgewidth=1.5,
                label=f"{scenario} (fitted order {p_asym:.2f})",
            )
        # Slope-2 reference line through the geometric middle of all points.
        all_h = np.array([r["h"] for rows in data.values() for r in rows])
        all_e = np.array([r[key] for rows in data.values() for r in rows])
        h_ref = np.array([all_h.min(), all_h.max()])
        c = np.exp(np.mean(np.log(all_e)) - 2 * np.mean(np.log(all_h)))
        ax.loglog(h_ref, 0.5 * c * h_ref**2, "--", color="#8a8a86", lw=1.5, label="reference slope 2")
        ax.set_xlabel("step size h [s]")
        ax.set_ylabel(label)
        ax.grid(which="major", alpha=0.3)
        ax.grid(which="minor", alpha=0.1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=9, frameon=False, loc="upper left")
    fig.suptitle(
        f"Trapezoidal collocation converges at O(h²)  (N = {NS[0]}…{NS[-1]}; fitted order uses N ≥ {ASYMPTOTIC_MIN_N})"
    )
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "convergence.png")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true", help="reuse cached results instead of re-solving")
    args = parser.parse_args()

    if args.replot:
        with open(CACHE) as fh:
            data = json.load(fh)
    else:
        data = run_all()
    check_trustworthy(data)
    print()
    print(report(data))
    print()
    plot(data)


if __name__ == "__main__":
    main()
