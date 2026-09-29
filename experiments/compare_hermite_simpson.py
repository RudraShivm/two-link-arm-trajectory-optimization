#!/usr/bin/env python3
"""Original vs Improved Hermite-Simpson Collocation: before/after comparison.

Same three experiments as experiments/compare_trapezoidal.py, run here for
Hermite-Simpson instead. Writes the results to experiments/results/:

  1. Speed vs grid size   (rest_to_rest, N = 10, 20, 40, 80)
     -> Is the improved solver faster, and does the gap grow with N?
  2. Ablation at N = 40   (turn on S1, S2, S3 one after another, then S5)
     -> How much does EACH improvement contribute?
  3. All scenarios at N = 30
     -> Does it still work (and give the same answer) on every task?

Every table also shows the COST and the constraint VIOLATION, to prove the
improved solver finds the same answer, only faster.

Note on N=80: the ORIGINAL Hermite-Simpson solver is considerably slower
than the original Trapezoidal solver at large N (defects touch 3 points
instead of 2, so its dense finite-difference Jacobian is bigger). At N=80
it can take several minutes for a single solve; this script times it once
(not the usual `--repeats` median) once N reaches `SLOW_LIMIT`.

Usage (from the project root):
    PYTHONPATH=. python3 experiments/compare_hermite_simpson.py            # full run (~15-20 min)
    PYTHONPATH=. python3 experiments/compare_hermite_simpson.py --quick    # smaller, ~2 min

Outputs:
    experiments/results/hermite_simpson_comparison.md   (all tables)
    experiments/results/hs_speed_vs_N.png                (plot for experiment 1)
    experiments/results/hs_ablation.png                  (plot for experiment 2)
"""

import argparse
import os
import statistics
import sys
import time

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np

from arm_opt.analysis.reality_check import perform_reality_check
from arm_opt.scenarios import SCENARIOS
from arm_opt.solvers.hermite_simpson import HermiteSimpsonCollocationSolver
from arm_opt.solvers.improved_hermite_simpson import ImprovedHermiteSimpsonSolver, solve_with_warm_start

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# -----------------------------------------------------------------------------
# Solver configurations
# -----------------------------------------------------------------------------
# Each entry: display name -> function(problem) -> TrajectoryResult.
# The ablation list is CUMULATIVE: each row adds one improvement on top of the
# previous row, so the time drop between two rows is that improvement's effect.
#
# Unlike Trapezoidal, there is no S4 (scaling) here yet: Hermite-Simpson's
# Simpson-quadrature weights are different from Trapezoidal's, so the scaling
# derivation would need its own analysis rather than a straight port. S1-S3
# and S5 (warm start) carry over unchanged.

ALL_OFF = dict(fix_endpoints_in_bounds=False, vectorized=False, sparse_jacobian=False)

ABLATION = [
    ("Original (hermite_simpson.py)", lambda pr: HermiteSimpsonCollocationSolver(pr).solve()),
    ("+ S1 endpoints as bounds", lambda pr: ImprovedHermiteSimpsonSolver(
        pr, **{**ALL_OFF, "fix_endpoints_in_bounds": True}).solve()),
    ("+ S2 vectorized physics", lambda pr: ImprovedHermiteSimpsonSolver(
        pr, **{**ALL_OFF, "fix_endpoints_in_bounds": True, "vectorized": True}).solve()),
    ("+ S3 sparse Jacobian", lambda pr: ImprovedHermiteSimpsonSolver(pr).solve()),
    ("+ S5 warm start", lambda pr: solve_with_warm_start(pr)),
]

# S1 (fix_endpoints_in_bounds) deliberately OFF, to test whether S1 was
# ever responsible for any slowdown (see experiments/compare_hs_no_s1.py
# for why this question came up).
NO_S1 = dict(fix_endpoints_in_bounds=False)

# The five configurations compared in experiments 1 and 3.
MAIN = [
    ("Original", lambda pr: HermiteSimpsonCollocationSolver(pr).solve()),
    ("Improved (S1-S3)", lambda pr: ImprovedHermiteSimpsonSolver(pr).solve()),
    ("Improved + warm start (S1-S3+S5)", lambda pr: solve_with_warm_start(pr)),
    ("S2+S3 only (no S1)", lambda pr: ImprovedHermiteSimpsonSolver(
        pr, **NO_S1, vectorized=True, sparse_jacobian=True).solve()),
    ("S2+S3+S5 (no S1)", lambda pr: solve_with_warm_start(pr, **NO_S1)),
]


# -----------------------------------------------------------------------------
# Running and measuring
# -----------------------------------------------------------------------------


def run(solve_fn, problem, repeats):
    """Runs a solver `repeats` times. Returns (last result, median solve time).

    We take the MEDIAN time because a single timing can be disturbed by
    whatever else the machine is doing at that moment.
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


def experiment_speed_vs_n(Ns, repeats, slow_repeats_limit, on_row=None):
    """Experiment 1: solve time vs N on rest_to_rest, for the three MAIN configs.

    Args:
        on_row: optional callback(rows_so_far, Ns) called after EVERY row, so
                the caller can save partial plots/reports without waiting for
                the whole experiment (or the whole script) to finish.
    """
    print("\n=== Experiment 1: speed vs N (rest_to_rest) ===", flush=True)
    rows = {}  # (N, name) -> row dict
    for N in Ns:
        problem = SCENARIOS["rest_to_rest"]().create_problem(n_nodes=N)
        baseline = None
        for name, fn in MAIN:
            # The original solver gets very slow at large N, so we time it only
            # once there (noted in the report).
            reps = 1 if (name == "Original" and N >= slow_repeats_limit) else repeats
            t_wall0 = time.perf_counter()
            res, t = run(fn, problem, reps)
            wall = time.perf_counter() - t_wall0
            r = row(name, problem, res, t, baseline)
            baseline = baseline or r
            rows[(N, name)] = r
            print(f"  N={N:3d} {name:34s} {t:7.2f}s  iters={res.iterations:4d}  "
                  f"cost={res.cost:.4f}  (wall {wall:.1f}s)", flush=True)
            if on_row is not None:
                on_row(dict(rows), N)
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
        print(f"  {name:30s} {t:7.2f}s  iters={res.iterations:4d}  cost={res.cost:.4f}", flush=True)
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
            print(f"  {key:14s} {name:34s} ok={res.success!s:5s} {t:7.2f}s iters={res.iterations:4d} "
                  f"cost={res.cost:.3f}", flush=True)
        out[key] = rows
    return out


# -----------------------------------------------------------------------------
# Plots (same palette/style as compare_trapezoidal.py, for visual consistency
# between the two reports)
# -----------------------------------------------------------------------------

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#9b59b6", "#c0392b"]
MARKERS = ["o", "s", "^", "D", "v"]
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
    end_times = []  # collect (time, index) so close-together labels can be un-stacked
    for i, (name, _) in enumerate(MAIN):
        times = [rows[(N, name)]["time"] for N in Ns]
        ax.plot(Ns, times, color=COLORS[i], marker=MARKERS[i], markersize=8, linewidth=2, label=name)
        end_times.append((times[-1], i))

    # Labels for lines that end close together get nudged apart so they
    # don't overlap illegibly. Walk the end values from smallest to
    # largest and enforce a minimum vertical gap in LOG space (since the
    # axis is log-scaled), pushing each label up past the previous one
    # if needed, rather than only comparing pairs independently.
    end_times.sort()
    min_gap = 0.42  # ~ minimum readable spacing between two 9pt labels here
    placed_log = []
    for t, i in end_times:
        logt = np.log(t)
        if placed_log and logt - placed_log[-1] < min_gap:
            logt = placed_log[-1] + min_gap
        placed_log.append(logt)
        y = np.exp(logt)
        ax.annotate(f"{t:.1f} s", (Ns[-1], y), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_yscale("log")
    ax.set_xticks(Ns)
    ax.set_xlabel("N (number of intervals)", color=INK)
    ax.set_ylabel("Solve time (s, log scale)", color=INK)
    ax.set_title("Hermite-Simpson collocation: solve time vs grid size (rest_to_rest)", color=INK,
                 fontsize=11, loc="left")
    ax.legend(frameon=False, labelcolor=INK)
    ax.set_xlim(Ns[0] - 3, Ns[-1] * 1.22)
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
    ax.set_title(f"Effect of each, applied cumulatively (rest_to_rest, N={N})",
                 color=INK, fontsize=11, loc="left")
    ax.set_xlim(0, max(times) * 1.3)
    ax.tick_params(axis="y", colors=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def write_report(args, Ns, repeats, slow_limit, ablation_N, scen_N, speed, ablation, scenarios):
    """Builds the markdown report from whatever has been computed SO FAR.

    Called after every experiment (and, for experiment 1, after every row),
    so a partial run still leaves a readable report and plots on disk
    instead of nothing at all until the very end.
    """
    lines = [
        "# Original vs Improved Hermite-Simpson Collocation",
        "",
        "Generated by `experiments/compare_hermite_simpson.py`"
        + (" (`--quick` mode)" if args.quick else "") + ".",
        "",
        "- **Speedup** = original time / this time (same N, same scenario).",
        "- **Cost diff vs original** = |cost − original cost| / original cost. Near 0 means the same answer.",
        "- **Max violation** = worst start/end/defect error (same definition for both solvers).",
        "- **Final drift** = end-effector miss distance when the torques are replayed in a",
        "  high-accuracy simulation (the benchmark's reality check).",
        f"- Times are the median of {repeats} run(s); the original solver at N ≥ {slow_limit} was timed once.",
        "- Warm-start times **include** the coarse-grid solves; its iterations are the total over all levels.",
        "- Unlike the Trapezoidal comparison, there is no S4 (scaling) row here yet;",
        "  S1, S2, S3 and S5 (warm start) are ported over, S4 is left as future work.",
        "",
        "## 1. Speed vs grid size (rest_to_rest)",
        "",
        "![speed vs N](hs_speed_vs_N.png)",
        "",
        table_header(with_n=True),
    ]
    any_speed_rows = False
    for N in Ns:
        for name, _ in MAIN:
            if (N, name) in speed:  # tolerate a run still in progress
                lines.append(fmt_row(speed[(N, name)], with_n=N))
                any_speed_rows = True
    if not any_speed_rows:
        lines.append("| *(still running...)* | | | | | | | | |")

    lines += [
        "",
        f"## 2. Ablation: each improvement's contribution (rest_to_rest, N={ablation_N})",
        "",
        "Each row adds one improvement on top of the row above it.",
        "",
    ]
    if ablation:
        lines += ["![ablation](hs_ablation.png)", "", table_header()]
        lines += [fmt_row(r) for r in ablation]
    else:
        lines.append("*(not reached yet)*")

    lines += ["", f"## 3. All scenarios (N={scen_N})", ""]
    if scenarios:
        for key, rows in scenarios.items():
            lines += [f"### {key}", "", table_header()]
            lines += [fmt_row(r) for r in rows]
            lines.append("")
    else:
        lines.append("*(not reached yet)*")

    report = os.path.join(RESULTS_DIR, "hermite_simpson_comparison.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="smaller grids, 1 repeat (~2 min)")
    parser.add_argument("--repeats", type=int, default=3, help="timing repeats (median is reported)")
    parser.add_argument(
        "--ns", type=int, nargs="+", default=None,
        help="grid sizes for experiment 1, e.g. --ns 10 20 40 (skip the slow N=80 point)",
    )
    args = parser.parse_args()

    if args.quick:
        Ns, repeats, ablation_N, scen_N = [10, 20, 40], 1, 20, 15
    else:
        Ns, repeats, ablation_N, scen_N = [10, 20, 40, 80], args.repeats, 40, 30
    if args.ns is not None:
        Ns = args.ns
    # The original Hermite-Simpson solver is slower than Trapezoidal's at the
    # same N (defects touch 3 points, not 2), so we stop repeating it earlier.
    slow_limit = 40

    os.makedirs(RESULTS_DIR, exist_ok=True)
    speed_path = os.path.join(RESULTS_DIR, "hs_speed_vs_N.png")
    ablation_path = os.path.join(RESULTS_DIR, "hs_ablation.png")

    # ---- Experiment 1: write the plot + report after EVERY row, not just at
    # the end, so a slow N=80 original solve does not block you from seeing
    # everything computed so far. ------------------------------------------
    def after_each_row(rows_so_far, N_done):
        completed_Ns = [N for N in Ns if all((N, name) in rows_so_far for name, _ in MAIN)]
        if completed_Ns:
            plot_speed_vs_n(rows_so_far, completed_Ns, speed_path)
        write_report(args, Ns, repeats, slow_limit, ablation_N, scen_N, rows_so_far, [], {})
        print(f"    (partial plot/report updated after N={N_done})", flush=True)

    speed = experiment_speed_vs_n(Ns, repeats, slow_limit, on_row=after_each_row)
    plot_speed_vs_n(speed, Ns, speed_path)
    write_report(args, Ns, repeats, slow_limit, ablation_N, scen_N, speed, [], {})
    print(f"\n(experiment 1 done -> {speed_path} and the report are up to date)", flush=True)

    # ---- Experiment 2 --------------------------------------------------------
    ablation = experiment_ablation(ablation_N, repeats)
    plot_ablation(ablation, ablation_N, ablation_path)
    write_report(args, Ns, repeats, slow_limit, ablation_N, scen_N, speed, ablation, {})
    print(f"(experiment 2 done -> {ablation_path} and the report are up to date)", flush=True)

    # ---- Experiment 3 --------------------------------------------------------
    scenarios = experiment_scenarios(scen_N)
    write_report(args, Ns, repeats, slow_limit, ablation_N, scen_N, speed, ablation, scenarios)

    report = os.path.join(RESULTS_DIR, "hermite_simpson_comparison.md")
    print(f"\nAll done. Final report: {report}")


if __name__ == "__main__":
    sys.exit(main())
