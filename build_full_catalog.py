#!/usr/bin/env python3
import json
import os

from arm_opt.scenarios import SCENARIOS
from arm_opt.visualization.web_export import build_scenario_dict
from arm_opt.solvers.shooting import ShootingSolver
from arm_opt.solvers.trapezoidal import TrapezoidalCollocationSolver
from arm_opt.solvers.hermite_simpson import HermiteSimpsonCollocationSolver


def solve_scenario(scenario, n_nodes, max_iter_shooting=100, max_iter_colloc=150):
    problem = scenario.create_problem(n_nodes=n_nodes)
    results = {}

    print("  shooting...")
    try:
        results["shooting"] = ShootingSolver(problem).solve(
            max_iter=max_iter_shooting, ftol=1e-5
        )
        print(
            f"    success={results['shooting'].success}, "
            f"cost={results['shooting'].cost:.2f}, "
            f"time={results['shooting'].solve_time:.2f}s"
        )
    except Exception as e:
        print(f"    error: {e}")

    print("  trapezoidal...")
    try:
        results["trapezoidal"] = TrapezoidalCollocationSolver(problem).solve(
            max_iter=max_iter_colloc, ftol=1e-5
        )
        print(
            f"    success={results['trapezoidal'].success}, "
            f"cost={results['trapezoidal'].cost:.2f}, "
            f"time={results['trapezoidal'].solve_time:.2f}s"
        )
    except Exception as e:
        print(f"    error: {e}")

    print("  hermite-simpson...")
    try:
        results["hermite_simpson"] = HermiteSimpsonCollocationSolver(problem).solve(
            max_iter=max_iter_colloc, ftol=1e-5
        )
        print(
            f"    success={results['hermite_simpson'].success}, "
            f"cost={results['hermite_simpson'].cost:.2f}, "
            f"time={results['hermite_simpson'].solve_time:.2f}s"
        )
    except Exception as e:
        print(f"    error: {e}")

    return problem, results


def main():
    catalog = {"active_scenario": "rest_to_rest", "scenarios": {}}

    configs = [
        ("rest_to_rest", 20, 100, 150),
        ("obstacle", 25, 120, 150),
        ("high_speed", 20, 80, 150),
        ("swing_up", 30, 50, 150),
    ]

    for key, n_nodes, max_s, max_c in configs:
        scenario = SCENARIOS[key]()
        print(f"\n{scenario.name} (N={n_nodes})")
        problem, results = solve_scenario(
            scenario, n_nodes, max_iter_shooting=max_s, max_iter_colloc=max_c
        )

        obs_spec = None
        if hasattr(scenario, "obs_center"):
            obs_spec = (tuple(scenario.obs_center), scenario.obs_radius)

        catalog["scenarios"][key] = build_scenario_dict(
            problem,
            results,
            obstacle_spec=obs_spec,
            scenario_name=scenario.name,
            scenario_desc=scenario.description,
        )

    os.makedirs("web_viewer", exist_ok=True)
    with open("web_viewer/catalog.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
    with open("web_viewer/trajectory_data.json", "w", encoding="utf-8") as f:
        json.dump(catalog["scenarios"]["rest_to_rest"], f, indent=2)

    print(f"\nWrote catalog with {len(catalog['scenarios'])} scenarios.")


if __name__ == "__main__":
    main()
