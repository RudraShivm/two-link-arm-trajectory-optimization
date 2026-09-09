#!/usr/bin/env python3
"""Generates all showcase artifacts: MP4 animations, comparative plots, and web viewer data."""

import argparse
import os
import sys

from arm_opt.scenarios import SCENARIOS
from arm_opt.visualization.animator import SynchronizedArmAnimator
from arm_opt.visualization.plots import plot_comparative_summary
from arm_opt.visualization.web_export import export_trajectory_json


OUTPUT_DIR = "output"


def generate_showcase(scenario_key: str, n_nodes: int = 30):
    """Runs solvers, exports plots, animation, and web JSON for a given scenario."""
    scenario_cls = SCENARIOS[scenario_key]
    scenario = scenario_cls()

    print(f"\n{'='*72}")
    print(f"  Generating showcase for: {scenario.name}")
    print(f"{'='*72}\n")

    problem = scenario.create_problem(n_nodes=n_nodes)
    results = scenario.run_comparison(n_nodes=n_nodes)

    converged = {k: v for k, v in results.items() if v.success}
    if not converged:
        print(f"  WARNING: No solver converged for {scenario.name}. Skipping showcase.\n")
        return

    for key, res in results.items():
        status = "CONVERGED" if res.success else "FAILED"
        print(f"  [{status}] {res.method_name}: cost={res.cost:.2f}, time={res.solve_time:.3f}s")

    obs_spec = None
    if scenario_key == "obstacle" and hasattr(scenario, "obs_center"):
        obs_spec = (tuple(scenario.obs_center), scenario.obs_radius)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Comparative 4-panel plot
    plot_path = os.path.join(OUTPUT_DIR, f"{scenario_key}_comparison.png")
    plot_comparative_summary(problem, results, obstacle_spec=obs_spec, save_path=plot_path)
    print(f"  [Saved] Comparative plot: {plot_path}")

    # 2. Synchronized MP4 animation (only with converged methods)
    mp4_path = os.path.join(OUTPUT_DIR, f"{scenario_key}_animation.mp4")
    try:
        animator = SynchronizedArmAnimator(
            problem, converged, obstacle_spec=obs_spec, fps=50
        )
        animator.render_video(mp4_path)
        print(f"  [Saved] Animation video: {mp4_path}")
    except Exception as e:
        print(f"  [Warning] Animation render failed: {e}")

    # 3. Web viewer JSON export
    json_path = os.path.join("web_viewer", f"trajectory_data_{scenario_key}.json")
    export_trajectory_json(problem, results, obstacle_spec=obs_spec, output_path=json_path)
    print(f"  [Saved] Web viewer JSON: {json_path}")

    # Also write the default trajectory_data.json for the web viewer
    default_json = os.path.join("web_viewer", "trajectory_data.json")
    export_trajectory_json(problem, results, obstacle_spec=obs_spec, output_path=default_json)
    print(f"  [Saved] Default web data: {default_json}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate showcase artifacts for Two-Link Arm Trajectory Optimization"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="rest_to_rest",
        choices=list(SCENARIOS.keys()) + ["all"],
        help="Scenario to generate showcase for (default: rest_to_rest)",
    )
    parser.add_argument(
        "--nodes", type=int, default=30, help="Number of collocation grid intervals"
    )
    args = parser.parse_args()

    if args.scenario == "all":
        for sc in SCENARIOS.keys():
            generate_showcase(sc, n_nodes=args.nodes)
    else:
        generate_showcase(args.scenario, n_nodes=args.nodes)

    print("Showcase generation complete.")
    print(f"Open web_viewer/index.html in a browser for the interactive demo.")


if __name__ == "__main__":
    main()

