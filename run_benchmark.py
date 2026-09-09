#!/usr/bin/env python3
"""Main comparative benchmark runner for the two-link robot arm trajectory optimization."""

import argparse
import sys
from tabulate import tabulate
import numpy as np

from arm_opt.scenarios import SCENARIOS
from arm_opt.analysis.metrics import compute_metrics
from arm_opt.analysis.reality_check import perform_reality_check
from arm_opt.visualization.plots import plot_comparative_summary
from arm_opt.visualization.web_export import export_trajectory_json


def run_benchmark(scenario_key: str, n_nodes: int = 30, plot: bool = True):
    if scenario_key not in SCENARIOS:
        print(f"Error: Unknown scenario '{scenario_key}'. Available: {list(SCENARIOS.keys())}")
        sys.exit(1)

    scenario_cls = SCENARIOS[scenario_key]
    scenario = scenario_cls()

    print("\n" + "=" * 80)
    print(f" BENCHMARK EXPERIMENT: {scenario.name.upper()}")
    print("=" * 80)
    print(f"Description:\n  {scenario.description}\n")

    problem = scenario.create_problem(n_nodes=n_nodes)
    print(f"Grid Discretization: {n_nodes} intervals (dt = {problem.dt:.4f}s), Horizon = {problem.duration:.2f}s")
    print("Executing solvers: [1] Shooting (RK4), [2] Trapezoidal, [3] Hermite-Simpson...\n")

    results = scenario.run_comparison(n_nodes=n_nodes)

    # Compile quantitative metrics table
    rows = []
    reality_drifts = {}

    for key, res in results.items():
        metrics = compute_metrics(problem, res)
        # Perform reality check replay if solver converged
        if res.success:
            rc = perform_reality_check(problem, res)
            reality_drifts[key] = rc
            max_drift_str = f"{rc.max_cartesian_drift:.4f} m"
            term_drift_str = f"{rc.terminal_cartesian_drift:.4f} m"
        else:
            max_drift_str = "N/A (diverged)"
            term_drift_str = "N/A (diverged)"

        rows.append([
            res.method_name,
            "SUCCESS" if res.success else "FAILED",
            f"{metrics.solve_time:.3f} s",
            metrics.iterations,
            f"{metrics.total_effort:.2f}",
            f"{metrics.peak_torque:.2f} N*m",
            f"{metrics.terminal_ee_error_meters:.5f} m",
            max_drift_str,
            term_drift_str,
        ])

    headers = [
        "Method",
        "Status",
        "Solve Time",
        "Iterations",
        "Effort ∫||τ||²",
        "Peak Torque",
        "Opt EE Err",
        "Sim Max Drift",
        "Sim Final Drift",
    ]

    try:
        from tabulate import tabulate
        print(tabulate(rows, headers=headers, tablefmt="github"))
    except ImportError:
        # Fallback simple printing if tabulate isn't installed
        header_str = " | ".join(headers)
        print(header_str)
        print("-" * len(header_str))
        for row in rows:
            print(" | ".join(str(x) for x in row))

    obs_spec = None
    if scenario_key == "obstacle" and hasattr(scenario, "obs_center"):
        obs_spec = (tuple(scenario.obs_center), scenario.obs_radius)

    if plot:
        plot_path = f"benchmark_{scenario_key}.png"
        plot_comparative_summary(problem, results, obstacle_spec=obs_spec, save_path=plot_path)
        print(f"\n[Artifact Saved] Comparative plot figure: {plot_path}")

    json_path = "web_viewer/trajectory_data.json"
    export_trajectory_json(problem, results, obstacle_spec=obs_spec, output_path=json_path)
    print(f"[Artifact Saved] Exported web visualizer data: {json_path}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Two-Link Arm Trajectory Optimization Benchmark")
    parser.add_argument(
        "--scenario",
        type=str,
        default="rest_to_rest",
        choices=list(SCENARIOS.keys()) + ["all"],
        help="Benchmark scenario to evaluate",
    )
    parser.add_argument("--nodes", type=int, default=30, help="Number of grid intervals N")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot generation")

    args = parser.parse_args()

    if args.scenario == "all":
        for sc in SCENARIOS.keys():
            run_benchmark(sc, n_nodes=args.nodes, plot=not args.no_plot)
    else:
        run_benchmark(args.scenario, n_nodes=args.nodes, plot=not args.no_plot)


if __name__ == "__main__":
    main()

