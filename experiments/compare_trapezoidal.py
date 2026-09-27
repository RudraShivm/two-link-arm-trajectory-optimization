#!/usr/bin/env python3
"""Original vs Improved Trapezoidal Collocation: before/after comparison.

Runs three experiments and writes the results to experiments/results/:

  1. Speed vs grid size   (rest_to_rest, N = 10, 20, 40, 80)
     -> Is the improved solver faster, and does the gap grow with N?
  2. Ablation at N = 40   (turn on S1, S2, S3, S4, S5 one after another)
     -> How much does EACH improvement contribute?
  3. All scenarios at N = 30
     -> Does it still work (and give the same answer) on every task?

Every table also shows the COST and the constraint VIOLATION, to prove the
improved solver finds the same answer, only faster.

Usage (from the project root):
    PYTHONPATH=. python3 experiments/compare_trapezoidal.py            # full run (~10 min)
    PYTHONPATH=. python3 experiments/compare_trapezoidal.py --quick    # smaller, ~2 min

Outputs:
    experiments/results/trapezoidal_comparison.md   (all tables)
    experiments/results/speed_vs_N.png              (plot for experiment 1)
    experiments/results/ablation.png                (plot for experiment 2)
"""

import argparse
import os
import statistics
import sys

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np

from arm_opt.analysis.reality_check import perform_reality_check
from arm_opt.scenarios import SCENARIOS
from arm_opt.solvers.improved_trapezoidal import ImprovedTrapezoidalSolver, solve_with_warm_start
from arm_opt.solvers.trapezoidal import TrapezoidalCollocationSolver

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# -----------------------------------------------------------------------------
# Solver configurations
# -----------------------------------------------------------------------------
# Each entry: display name -> function(problem) -> TrajectoryResult.
# The ablation list is CUMULATIVE: each row adds one improvement on top of the
# previous row, so the time drop between two rows is that improvement's effect.

ALL_OFF = dict(fix_endpoints_in_bounds=False, vectorized=False, sparse_jacobian=False, scaling=False)

ABLATION = [
    ("Original (trapezoidal.py)", lambda pr: TrapezoidalCollocationSolver(pr).solve()),
    ("+ S1 endpoints as bounds", lambda pr: ImprovedTrapezoidalSolver(
        pr, **{**ALL_OFF, "fix_endpoints_in_bounds": True}).solve()),
    ("+ S2 vectorized physics", lambda pr: ImprovedTrapezoidalSolver(
        pr, **{**ALL_OFF, "fix_endpoints_in_bounds": True, "vectorized": True}).solve()),
    ("+ S3 sparse Jacobian", lambda pr: ImprovedTrapezoidalSolver(pr, scaling=False).solve()),
    ("+ S4 scaling", lambda pr: ImprovedTrapezoidalSolver(pr).solve()),
    ("+ S5 warm start", lambda pr: solve_with_warm_start(pr)),
]

# The three configurations compared in experiments 1 and 3.
MAIN = [
    ("Original", lambda pr: TrapezoidalCollocationSolver(pr).solve()),
    ("Improved (S1-S4)", lambda pr: ImprovedTrapezoidalSolver(pr).solve()),
    ("Improved + warm start (S1-S5)", lambda pr: solve_with_warm_start(pr)),
]


# -----------------------------------------------------------------------------
# Running and measuring
# -----------------------------------------------------------------------------


def run(solve_fn, problem, repeats):
    """Runs a solver `repeats` times. Returns (last result, median solve time).

    We take the MEDIAN time because a single timing can be disturbed by
    whatever else the laptop is doing at that moment.
    """
    times, res = [], None
    for _ in range(repeats):
        res = solve_fn(problem)
        times.append(res.solve_time)
    return res, statistics.median(times)


def final_drift(problem, res):
    """End-effector miss distance when the torques are replayed in a high-accuracy
    simulation (same 'reality check' the benchmark uses). None if the solve failed."""
    if not res.success:
        return None
    return perform_reality_check(problem, res).terminal_cartesian_drift


def row(name, problem, res, t, baseline=None):
    """Collects one table row as a dict."""
    r = {
        "name": name,
        "success": res.success,
        "time": t,
        "iters": res.iterations,
        "cost": res.cost,
        "viol": res.max_constraint_violation,
        "drift": final_drift(problem, res),
    }
    if baseline is not None:
        r["speedup"] = baseline["time"] / t
        # Relative cost difference to the original: ~0 means "same answer".
        r["cost_diff"] = abs(res.cost - baseline["cost"]) / abs(baseline["cost"])
    return r


def fmt_row(r, with_n=None):
    drift = "—" if r["drift"] is None else f"{r['drift']:.4f} m"
    speed = f"**{r['speedup']:.1f}×**" if "speedup" in r else "1.0×"
    cdiff = f"{r['cost_diff']:.1e}" if "cost_diff" in r else "—"
    cells = [r["name"], "✅" if r["success"] else "❌", f"{r['time']:.2f} s", speed,
             str(r["iters"]), f"{r['cost']:.3f}", cdiff, f"{r['viol']:.1e}", drift]
    if with_n is not None:
        cells.insert(0, str(with_n))
    return "| " + " | ".join(cells) + " |"


HEADER = ["Solver", "OK", "Time", "Speedup", "Iterations", "Cost", "Cost diff vs original",
          "Max violation", "Final drift"]


def table_header(with_n=False):
    h = (["N"] if with_n else []) + HEADER
    return "| " + " | ".join(h) + " |\n|" + "---|" * len(h)


# -----------------------------------------------------------------------------
# Experiments
# -----------------------------------------------------------------------------


def experiment_speed_vs_n(Ns, repeats, slow_repeats_limit):
    """Experiment 1: solve time vs N on rest_to_rest, for the three MAIN configs."""
    print("\n=== Experiment 1: speed vs N (rest_to_rest) ===", flush=True)
    rows = {}  # (N, name) -> row dict
    for N in Ns:
        problem = SCENARIOS["rest_to_rest"]().create_problem(n_nodes=N)
        baseline = None
        for name, fn in MAIN:
            # The original solver gets very slow at large N, so we time it only
            # once there (noted in the report).
            reps = 1 if (name == "Original" and N >= slow_repeats_limit) else repeats
            res, t = run(fn, problem, reps)
            r = row(name, problem, res, t, baseline)
            baseline = baseline or r
            rows[(N, name)] = r
            print(f"  N={N:3d} {name:32s} {t:7.2f}s  iters={res.iterations:4d}  cost={res.cost:.4f}",
                  flush=True)
    return rows


def experiment_ablation(N, repeats):
    """Experiment 2: turn improvements on one by one at a fixed N."""
    print(f"\n=== Experiment 2: ablation (rest_to_rest, N={N}) ===", flush=True)
    problem = SCENARIOS["rest_to_rest"]().create_problem(n_nodes=N)
    rows, baseline = [], None
    for name, fn in ABLATION:
        res, t = run(fn, problem, repeats)
        r = row(name, problem, res, t, baseline)
        baseline = baseline or r
        rows.append(r)
        print(f"  {name:28s} {t:7.2f}s  iters={res.iterations:4d}  cost={res.cost:.4f}", flush=True)
    return rows


def experiment_scenarios(N):
    """Experiment 3: every scenario, three MAIN configs, one run each."""
    print(f"\n=== Experiment 3: all scenarios (N={N}) ===", flush=True)
    out = {}
    for key, cls in SCENARIOS.items():
        problem = cls().create_problem(n_nodes=N)
        rows, baseline = [], None
        for name, fn in MAIN:
            res, t = run(fn, problem, 1)
            r = row(name, problem, res, t, baseline)
            baseline = baseline or r
            rows.append(r)
            print(f"  {key:14s} {name:32s} ok={res.success!s:5s} {t:7.2f}s iters={res.iterations:4d} "
                  f"cost={res.cost:.3f}", flush=True)
        out[key] = rows
    return out


# -----------------------------------------------------------------------------
# Plots (palette: validated categorical slots 1-3; marker shapes + direct
# labels as a second cue so the lines are not told apart by colour alone)
# -----------------------------------------------------------------------------

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_speed_vs_n(rows, Ns, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor=SURFACE)
    _style(ax)
    for i, (name, _) in enumerate(MAIN):
        times = [rows[(N, name)]["time"] for N in Ns]
        ax.plot(Ns, times, color=COLORS[i], marker=MARKERS[i], markersize=8, linewidth=2, label=name)
        # direct label at the right end of each line
        ax.annotate(f"{times[-1]:.1f} s", (Ns[-1], times[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(Ns)
    ax.set_xlabel("N (number of intervals)", color=INK)
    ax.set_ylabel("Solve time (s, log scale)", color=INK)
    ax.set_title("Trapezoidal collocation: solve time vs grid size (rest_to_rest)", color=INK,
                 fontsize=11, loc="left")
    ax.legend(frameon=False, labelcolor=INK)
    ax.set_xlim(Ns[0] - 3, Ns[-1] * 1.12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_ablation(rows, N, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.0), facecolor=SURFACE)
    _style(ax)
    ax.grid(True, axis="x", color=GRID)
    ax.grid(False, axis="y")
    names = [r["name"] for r in rows][::-1]  # top-to-bottom = order applied
    times = [r["time"] for r in rows][::-1]
    ax.barh(names, times, color=COLORS[0], height=0.6)
    for y, r in enumerate(rows[::-1]):
        label = f"{r['time']:.2f} s" + (f"  ({r['speedup']:.1f}×)" if "speedup" in r else "")
        ax.annotate(label, (r["time"], y), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9)
    ax.set_xlabel("Solve time (s)", color=INK)
    ax.set_title(f"Effect of each improvement, applied cumulatively (rest_to_rest, N={N})",
                 color=INK, fontsize=11, loc="left")
    ax.set_xlim(0, max(times) * 1.3)
    ax.tick_params(axis="y", colors=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="smaller grids, 1 repeat (~2 min)")
    parser.add_argument("--repeats", type=int, default=3, help="timing repeats (median is reported)")
    args = parser.parse_args()

    if args.quick:
        Ns, repeats, ablation_N, scen_N = [10, 20, 40], 1, 20, 20
    else:
        Ns, repeats, ablation_N, scen_N = [10, 20, 40, 80], args.repeats, 40, 30
    slow_limit = 80  # original solver is timed only once at N >= 80

    os.makedirs(RESULTS_DIR, exist_ok=True)
    speed = experiment_speed_vs_n(Ns, repeats, slow_limit)
    ablation = experiment_ablation(ablation_N, repeats)
    scenarios = experiment_scenarios(scen_N)

    plot_speed_vs_n(speed, Ns, os.path.join(RESULTS_DIR, "speed_vs_N.png"))
    plot_ablation(ablation, ablation_N, os.path.join(RESULTS_DIR, "ablation.png"))

    # ---- write the markdown report ------------------------------------------
    lines = [
        "# Original vs Improved Trapezoidal Collocation",
        "",
        "Generated by `experiments/compare_trapezoidal.py`"
        + (" (`--quick` mode)" if args.quick else "") + ".",
        "",
        "- **Speedup** = original time / this time (same N, same scenario).",
        "- **Cost diff vs original** = |cost − original cost| / original cost. Near 0 means the same answer.",
        "- **Max violation** = worst start/end/defect error (same definition for both solvers).",
        "- **Final drift** = end-effector miss distance when the torques are replayed in a",
        "  high-accuracy simulation (the benchmark's reality check).",
        f"- Times are the median of {repeats} run(s); the original solver at N ≥ {slow_limit} was timed once.",
        "- Warm-start times **include** the coarse-grid solves; its iterations are the total over all levels.",
        "",
        "## 1. Speed vs grid size (rest_to_rest)",
        "",
        "![speed vs N](speed_vs_N.png)",
        "",
        table_header(with_n=True),
    ]
    for N in Ns:
        for name, _ in MAIN:
            lines.append(fmt_row(speed[(N, name)], with_n=N))
    lines += [
        "",
        f"## 2. Ablation: each improvement's contribution (rest_to_rest, N={ablation_N})",
        "",
        "Each row adds one improvement on top of the row above it.",
        "",
        "![ablation](ablation.png)",
        "",
        table_header(),
    ]
    lines += [fmt_row(r) for r in ablation]
    lines += ["", f"## 3. All scenarios (N={scen_N})", ""]
    for key, rows in scenarios.items():
        lines += [f"### {key}", "", table_header()]
        lines += [fmt_row(r) for r in rows]
        lines.append("")

    report = os.path.join(RESULTS_DIR, "trapezoidal_comparison.md")
    with open(report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport written to {report}")


if __name__ == "__main__":
    sys.exit(main())
