"""A3: whole-arm obstacle check.

Part 1: the ORIGINAL scenario only checks the hand. Show that its links pass
        through the obstacle, and that a whole-arm check makes it infeasible.
Part 2: on the new obstacle, compare hand-only vs whole-arm check over several
        N and both cold and warm start. Save before/after pictures.

Clearance is measured on the A1 quadratic interpolant at 3000 time samples, so
it also sees what happens between nodes (negative = inside the obstacle).

Run:  PYTHONPATH=. python3 experiments/obstacle_checks.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from arm_opt.dynamics.kinematics import forward_kinematics
from arm_opt.scenarios.obstacle import ObstacleScenario
from arm_opt.solvers.improved_trapezoidal import ImprovedTrapezoidalSolver, solve_with_warm_start
from arm_opt.solvers.trapezoidal_utils import link_clearances, trapezoidal_interpolate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
NEW_OBSTACLE = dict(
    obstacle_center=ObstacleScenario.FULL_BODY_CENTER,
    obstacle_radius=ObstacleScenario.FULL_BODY_RADIUS,
)
NS = [20, 30, 40]
PICTURE_N = 30
N_DENSE = 3000

COLOR_LINK1, COLOR_LINK2 = "#2a78d6", "#eb6834"
COLOR_OK, COLOR_HIT = "#2a78d6", "#e34948"
COLOR_OBSTACLE = "#8a8a86"


def solve(scenario, n, warm):
    problem = scenario.create_problem(n_nodes=n)
    if warm:
        result = solve_with_warm_start(problem, max_iter=1000, ftol=1e-8)
    else:
        result = ImprovedTrapezoidalSolver(problem).solve(max_iter=1000, ftol=1e-8)
    return problem, result


def dense_clearance(problem, result, scenario):
    t = np.linspace(0.0, problem.duration, N_DENSE)
    x, _ = trapezoidal_interpolate(problem.arm, result.time, result.state, result.control, t)
    return t, x, link_clearances(problem.arm_params, x, scenario.obs_center, scenario.obs_radius)


def arm_points(params, q):
    elbow, hand = forward_kinematics(q, params)
    return np.array([[0.0, 0.0], elbow, hand])


def draw_arm_frames(ax, problem, result, scenario, title):
    """Stroboscopic picture: the arm at every node, red where any link is inside."""
    ax.add_patch(plt.Circle(scenario.obs_center, scenario.obs_radius, color=COLOR_OBSTACLE, alpha=0.35, lw=0))
    ax.add_patch(plt.Circle(scenario.obs_center, scenario.obs_radius, fill=False, color=COLOR_OBSTACLE, lw=1.5))
    c = link_clearances(problem.arm_params, result.state, scenario.obs_center, scenario.obs_radius)
    inside = np.minimum(c["link1"], c["link2"]) < 0
    for k, q in enumerate(result.state[:, :2]):
        pts = arm_points(problem.arm_params, q)
        color, alpha, lw = (COLOR_HIT, 0.9, 2.0) if inside[k] else (COLOR_OK, 0.35, 1.2)
        ax.plot(pts[:, 0], pts[:, 1], "-", color=color, alpha=alpha, lw=lw, solid_capstyle="round")
        ax.plot(pts[1:, 0], pts[1:, 1], "o", color=color, alpha=alpha, ms=3)
    hands = np.array([arm_points(problem.arm_params, q)[2] for q in result.state[:, :2]])
    ax.plot(hands[:, 0], hands[:, 1], ":", color="#3d3d3a", lw=1, label="hand path")
    ax.plot([0], [0], "ks", ms=6)
    ax.plot([], [], "-", color=COLOR_OK, lw=1.5, label="arm at a node")
    ax.plot([], [], "-", color=COLOR_HIT, lw=2, label=f"link inside obstacle ({inside.sum()} nodes)")
    ax.set_aspect("equal")
    ax.set_xlim(-0.3, 2.2)
    ax.set_ylim(-1.7, 1.7)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def draw_clearance(ax, t, clear, title):
    ax.axhspan(min(-0.35, clear["link2"].min() - 0.02), 0, color=COLOR_HIT, alpha=0.08, lw=0)
    ax.axhline(0, color="#3d3d3a", lw=1)
    ax.plot(t, clear["link1"], color=COLOR_LINK1, lw=2, label="link 1")
    ax.plot(t, clear["link2"], color=COLOR_LINK2, lw=2, label="link 2 (incl. hand)")
    ax.text(t[-1], -0.02, "inside obstacle", ha="right", va="top", fontsize=8, color=COLOR_HIT)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("clearance [m]")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def part1_original():
    print("## Part 1: original obstacle (1.2, 0), r = 0.35, hand-only check\n")
    scenario = ObstacleScenario()
    problem, result = solve(scenario, PICTURE_N, warm=True)
    t, _, clear = dense_clearance(problem, result, scenario)
    c_nodes = link_clearances(problem.arm_params, result.state, scenario.obs_center, scenario.obs_radius)
    print(f"solver success={result.success}, cost={result.cost:.1f}")
    print(f"min clearance: link1 {clear['link1'].min():+.3f} m, link2 {clear['link2'].min():+.3f} m, "
          f"hand {clear['hand'].min():+.3f} m (hand at nodes {c_nodes['hand'].min():+.3f} m)")
    inside = np.minimum(c_nodes["link1"], c_nodes["link2"]) < 0
    print(f"nodes with a link inside the obstacle: {inside.sum()} of {len(inside)}")

    full = ObstacleScenario(check_full_body=True)
    p_full, r_full = solve(full, PICTURE_N, warm=True)
    print(f"same obstacle WITH whole-arm check: success={r_full.success}, "
          f"max violation={r_full.max_constraint_violation:.2f} (infeasible: link 1 must cross x-axis)\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    draw_arm_frames(axes[0], problem, result, scenario, f"Original scenario (hand-only check), N = {PICTURE_N}")
    draw_clearance(axes[1], t, clear, "Clearance of each link over time")
    fig.suptitle("Bug: the hand avoids the obstacle, but both links pass through it")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "obstacle_bug_original.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}\n")
    return result, r_full


def part2_new_obstacle():
    c, r = NEW_OBSTACLE["obstacle_center"], NEW_OBSTACLE["obstacle_radius"]
    print(f"## Part 2: new obstacle {c}, r = {r}\n")
    lines = [
        "| N | start | check | success | cost | min link clearance (dense) | solve time |",
        "|---|---|---|---|---|---|---|",
    ]
    runs = {}
    for n in NS:
        for warm in (False, True):
            for full_body in (False, True):
                scenario = ObstacleScenario(**NEW_OBSTACLE, check_full_body=full_body)
                problem, result = solve(scenario, n, warm)
                _, _, clear = dense_clearance(problem, result, scenario)
                worst = min(clear["link1"].min(), clear["link2"].min())
                runs[(n, warm, full_body)] = (problem, result, scenario)
                lines.append(
                    f"| {n} | {'warm' if warm else 'cold'} | {'whole arm' if full_body else 'hand only'} | "
                    f"{result.success} | {result.cost:.1f} | {worst:+.3f} m{' ❌' if worst < -1e-3 else ''} | "
                    f"{result.solve_time:.1f} s |"
                )
                print(lines[-1], flush=True)
    table = "\n".join(lines)

    # Before/after picture: cold start (as the benchmark runs the solver).
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for col, full_body in enumerate((False, True)):
        problem, result, scenario = runs[(PICTURE_N, False, full_body)]
        t, _, clear = dense_clearance(problem, result, scenario)
        name = "whole-arm check (A3)" if full_body else "hand-only check (old)"
        draw_arm_frames(axes[0, col], problem, result, scenario, f"{name}, N = {PICTURE_N}, cost {result.cost:.1f}")
        draw_clearance(axes[1, col], t, clear, f"Clearance over time: {name}")
    fig.suptitle(f"Obstacle at {c}, r = {r}: before vs after the whole-arm check (cold start)")
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "obstacle_full_body.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nsaved {out}")
    return table


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    part1_original()
    table = part2_new_obstacle()
    print("\n" + table)


if __name__ == "__main__":
    main()
