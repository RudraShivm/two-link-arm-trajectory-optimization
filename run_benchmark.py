#!/usr/bin/env python3
import argparse
import sys

from arm_opt.scenarios import SCENARIOS
from arm_opt.analysis.metrics import compute_metrics
from arm_opt.analysis.reality_check import perform_reality_check
from arm_opt.visualization.plots import plot_comparative_summary
from arm_opt.visualization.web_export import export_trajectory_json


def run_benchmark(scenario_key: str, n_nodes: int = 30, plot: bool = True):
    if scenario_key not in SCENARIOS:
        print(f"Unknown scenario '{scenario_key}'. Available: {list(SCENARIOS.keys())}")
        sys.exit(1)

    scenario = SCENARIOS[scenario_key]()
    print(f"\n=== {scenario.name} ===")
    print(scenario.description)
    problem = scenario.create_problem(n_nodes=n_nodes)
    print(f"N={n_nodes}, dt={problem.dt:.4f}s, T={problem.duration:.2f}s\n")

    results = scenario.run_comparison(n_nodes=n_nodes)
    rows = []

    for key, res in results.items():
        metrics = compute_metrics(problem, res)
        if res.success:
            rc = perform_reality_check(problem, res)
            max_drift_str = f"{rc.max_cartesian_drift:.4f} m"
            term_drift_str = f"{rc.terminal_cartesian_drift:.4f} m"
        else:
            max_drift_str = "N/A"
            term_drift_str = "N/A"

        rows.append(
            [
                res.method_name,
                "OK" if res.success else "FAIL",
                f"{metrics.solve_time:.3f}s",
                metrics.iterations,
                f"{metrics.total_effort:.2f}",
                f"{metrics.peak_torque:.2f}",
                f"{metrics.terminal_ee_error_meters:.5f}",
                max_drift_str,
                term_drift_str,
            ]
        )

    headers = [
        "Method",
        "Status",
        "Time",
        "Iters",
        "Effort",
        "Peak tau",
        "EE err",
        "Max drift",
        "Final drift",
    ]

    try:
        from tabulate import tabulate

        print(tabulate(rows, headers=headers, tablefmt="github"))
    except ImportError:
        print(" | ".join(headers))
        print("-" * 72)
        for row in rows:
            print(" | ".join(str(x) for x in row))

    obs_spec = None
    if scenario_key == "obstacle" and hasattr(scenario, "obs_center"):
        obs_spec = (tuple(scenario.obs_center), scenario.obs_radius)

    if plot:
        plot_path = f"benchmark_{scenario_key}.png"
        plot_comparative_summary(problem, results, obstacle_spec=obs_spec, save_path=plot_path)
        print(f"Saved plot: {plot_path}")

    json_path = "web_viewer/trajectory_data.json"
    export_trajectory_json(
        problem,
        results,
        obstacle_spec=obs_spec,
        scenario_name=scenario.name,
        scenario_desc=scenario.description,
        output_path=json_path,
    )
    print(f"Saved JSON: {json_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Two-link arm trajectory optimization benchmark")
    parser.add_argument(
        "--scenario",
        type=str,
        default="rest_to_rest",
        choices=list(SCENARIOS.keys()) + ["all"],
    )
    parser.add_argument("--nodes", type=int, default=30)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    if args.scenario == "all":
        for sc in SCENARIOS.keys():
            run_benchmark(sc, n_nodes=args.nodes, plot=not args.no_plot)
    else:
        run_benchmark(args.scenario, n_nodes=args.nodes, plot=not args.no_plot)


if __name__ == "__main__":
    main()
