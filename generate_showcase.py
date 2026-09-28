#!/usr/bin/env python3
import argparse
import os

from arm_opt.scenarios import SCENARIOS
from arm_opt.visualization.animator import SynchronizedArmAnimator
from arm_opt.visualization.plots import plot_comparative_summary
from arm_opt.visualization.web_export import export_trajectory_json


OUTPUT_DIR = "output"


def generate_showcase(scenario_key: str, n_nodes: int = 30):
    scenario = SCENARIOS[scenario_key]()
    print(f"\n=== Showcase: {scenario.name} ===")

    problem = scenario.create_problem(n_nodes=n_nodes)
    results = scenario.run_comparison(n_nodes=n_nodes)
    converged = {k: v for k, v in results.items() if v.success}

    if not converged:
        print(f"No solver converged for {scenario.name}.")
        return

    for res in results.values():
        status = "OK" if res.success else "FAIL"
        print(f"  [{status}] {res.method_name}: cost={res.cost:.2f}, time={res.solve_time:.3f}s")

    obs_spec = None
    if scenario_key == "obstacle" and hasattr(scenario, "obs_center"):
        obs_spec = (tuple(scenario.obs_center), scenario.obs_radius)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plot_path = os.path.join(OUTPUT_DIR, f"{scenario_key}_comparison.png")
    plot_comparative_summary(problem, results, obstacle_spec=obs_spec, save_path=plot_path)
    print(f"  Saved {plot_path}")

    mp4_path = os.path.join(OUTPUT_DIR, f"{scenario_key}_animation.mp4")
    try:
        animator = SynchronizedArmAnimator(
            problem, converged, obstacle_spec=obs_spec, fps=50
        )
        animator.render_video(mp4_path)
        print(f"  Saved {mp4_path}")
    except Exception as e:
        print(f"  Animation failed: {e}")

    json_path = os.path.join("web_viewer", f"trajectory_data_{scenario_key}.json")
    export_trajectory_json(
        problem,
        results,
        obstacle_spec=obs_spec,
        scenario_name=scenario.name,
        scenario_desc=scenario.description,
        output_path=json_path,
    )
    export_trajectory_json(
        problem,
        results,
        obstacle_spec=obs_spec,
        scenario_name=scenario.name,
        scenario_desc=scenario.description,
        output_path=os.path.join("web_viewer", "trajectory_data.json"),
    )
    print(f"  Saved {json_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        default="rest_to_rest",
        choices=list(SCENARIOS.keys()) + ["all"],
    )
    parser.add_argument("--nodes", type=int, default=30)
    args = parser.parse_args()

    if args.scenario == "all":
        for sc in SCENARIOS.keys():
            generate_showcase(sc, n_nodes=args.nodes)
    else:
        generate_showcase(args.scenario, n_nodes=args.nodes)


if __name__ == "__main__":
    main()
