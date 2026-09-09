#!/usr/bin/env python3
"""Builds a complete multi-scenario catalog for the rich research console."""

import json
import os
import sys

from arm_opt.scenarios import SCENARIOS
from arm_opt.visualization.web_export import build_scenario_dict
from arm_opt.solvers.shooting import ShootingSolver
from arm_opt.solvers.trapezoidal import TrapezoidalCollocationSolver
from arm_opt.solvers.hermite_simpson import HermiteSimpsonCollocationSolver


def solve_scenario_safely(scenario, n_nodes, max_iter_shooting=100, max_iter_colloc=150):
    problem = scenario.create_problem(n_nodes=n_nodes)
    results = {}

    print(f"    [1/3] Solving Shooting (RK4)...")
    try:
        s_solver = ShootingSolver(problem)
        results["shooting"] = s_solver.solve(max_iter=max_iter_shooting, ftol=1e-5)
        print(f"      -> Shooting success={results['shooting'].success}, cost={results['shooting'].cost:.2f}, time={results['shooting'].solve_time:.2f}s")
    except Exception as e:
        print(f"      -> Shooting error: {e}")

    print(f"    [2/3] Solving Trapezoidal Collocation...")
    try:
        t_solver = TrapezoidalCollocationSolver(problem)
        results["trapezoidal"] = t_solver.solve(max_iter=max_iter_colloc, ftol=1e-5)
        print(f"      -> Trapezoidal success={results['trapezoidal'].success}, cost={results['trapezoidal'].cost:.2f}, time={results['trapezoidal'].solve_time:.2f}s")
    except Exception as e:
        print(f"      -> Trapezoidal error: {e}")

    print(f"    [3/3] Solving Hermite-Simpson Collocation...")
    try:
        h_solver = HermiteSimpsonCollocationSolver(problem)
        results["hermite_simpson"] = h_solver.solve(max_iter=max_iter_colloc, ftol=1e-5)
        print(f"      -> Hermite-Simpson success={results['hermite_simpson'].success}, cost={results['hermite_simpson'].cost:.2f}, time={results['hermite_simpson'].solve_time:.2f}s")
    except Exception as e:
        print(f"      -> Hermite-Simpson error: {e}")

    return problem, results


def main():
    print("=" * 76)
    print("  Building Multi-Scenario Research Console Catalog (5 Benchmark Scenarios)")
    print("=" * 76)

    catalog = {
        "active_scenario": "rest_to_rest",
        "scenarios": {},
    }

    scenario_configs = [
        ("rest_to_rest", 20, 100, 150),
        ("obstacle", 25, 120, 150),
        ("high_speed", 20, 80, 150),
        ("under_torqued", 18, 50, 150),
        ("reversal", 20, 60, 150),
    ]

    for key, n_nodes, max_s, max_c in scenario_configs:
        scenario_cls = SCENARIOS[key]
        scenario = scenario_cls()
        print(f"\n>>> Compiling: {scenario.name} (nodes={n_nodes})...")

        problem, results = solve_scenario_safely(scenario, n_nodes, max_iter_shooting=max_s, max_iter_colloc=max_c)

        obs_spec = None
        if hasattr(scenario, "obs_center"):
            obs_spec = (tuple(scenario.obs_center), scenario.obs_radius)

        scenario_dict = build_scenario_dict(
            problem,
            results,
            obstacle_spec=obs_spec,
            scenario_name=scenario.name,
            scenario_desc=scenario.description,
        )

        catalog["scenarios"][key] = scenario_dict
        print(f"  [OK] Successfully registered {key} ({scenario.name}) into catalog")

    os.makedirs("web_viewer", exist_ok=True)
    catalog_path = "web_viewer/catalog.json"
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    default_path = "web_viewer/trajectory_data.json"
    with open(default_path, "w", encoding="utf-8") as f:
        json.dump(catalog["scenarios"]["rest_to_rest"], f, indent=2)

    print(f"\n[Complete] Master catalog saved to {catalog_path} with {len(catalog['scenarios'])} scenarios")
    print(f"[Complete] Default scenario saved to {default_path}")


if __name__ == "__main__":
    main()
