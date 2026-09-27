"""A1: linear vs quadratic (trapezoidal-consistent) state interpolation.

Prints a table of reality-check drift for both interpolations and saves a plot
of one joint angle between nodes.

Run:  PYTHONPATH=. python3 experiments/trapezoidal_interpolation.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from arm_opt.analysis.reality_check import perform_reality_check
from arm_opt.scenarios import SCENARIOS
from arm_opt.solvers.improved_trapezoidal import solve_with_warm_start
from arm_opt.solvers.trapezoidal_utils import make_state_interp_fn, trapezoidal_interpolate

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SCENARIO = "rest_to_rest"
NS = [10, 20, 40]
PLOT_N = 10  # coarse grid, so the difference between the curves is visible


def solve(n):
    problem = SCENARIOS[SCENARIO]().create_problem(n_nodes=n)
    result = solve_with_warm_start(problem, ftol=1e-10)
    return problem, result


def local_interpolation_error(problem, result, samples_per_interval=20):
    """Max ||x_true(t) - x_interp(t)|| where x_true restarts from the node at every interval.

    Restarting removes the drift that builds up over the whole trajectory, so this
    measures only how well each interpolant describes the motion inside one interval.
    """
    arm, t, X, U = problem.arm, result.time, result.state, result.control
    err_lin, err_quad = 0.0, 0.0
    for k in range(len(t) - 1):
        t_k = np.linspace(t[k], t[k + 1], samples_per_interval + 1)

        def rhs(tt, x):
            u = U[k] + (tt - t[k]) / (t[k + 1] - t[k]) * (U[k + 1] - U[k])  # linear torque
            return arm.state_derivative(x, u)

        sol = solve_ivp(rhs, (t[k], t[k + 1]), X[k], method="DOP853", t_eval=t_k, rtol=1e-10, atol=1e-10)
        x_true = sol.y.T
        x_quad, _ = trapezoidal_interpolate(arm, t, X, U, t_k)
        s = ((t_k - t[k]) / (t[k + 1] - t[k]))[:, None]
        x_lin = X[k] + s * (X[k + 1] - X[k])
        err_lin = max(err_lin, np.max(np.linalg.norm(x_true - x_lin, axis=1)))
        err_quad = max(err_quad, np.max(np.linalg.norm(x_true - x_quad, axis=1)))
    return err_lin, err_quad


def main():
    rows = []
    for n in NS:
        problem, result = solve(n)
        rc_lin = perform_reality_check(problem, result)
        rc_quad = perform_reality_check(problem, result, state_interp_fn=make_state_interp_fn(problem, result))
        loc_lin, loc_quad = local_interpolation_error(problem, result)
        rows.append((n, result.success, rc_lin, rc_quad, loc_lin, loc_quad))

    lines = [
        f"### A1: linear vs quadratic interpolation ({SCENARIO}, warm start, ftol=1e-10)",
        "",
        "Reality check (one open-loop simulation over the whole motion):",
        "",
        "| N | success | max state drift (linear) | max state drift (quadratic) | "
        "max EE drift (linear) | max EE drift (quadratic) | final drift (both) |",
        "|---|---|---|---|---|---|---|",
    ]
    for n, ok, lin, quad, _, _ in rows:
        assert lin.terminal_cartesian_drift == quad.terminal_cartesian_drift
        lines.append(
            f"| {n} | {ok} | {lin.max_state_drift:.4f} | {quad.max_state_drift:.4f} | "
            f"{lin.max_cartesian_drift:.4f} m | {quad.max_cartesian_drift:.4f} m | "
            f"{lin.terminal_cartesian_drift:.4f} m |"
        )
    lines += [
        "",
        "Local interpolation error (simulation restarted from the node at every interval):",
        "",
        "| N | linear | quadratic | linear / quadratic |",
        "|---|---|---|---|",
    ]
    for n, _, _, _, loc_lin, loc_quad in rows:
        lines.append(f"| {n} | {loc_lin:.4f} | {loc_quad:.4f} | {loc_lin / loc_quad:.1f}x |")
    table = "\n".join(lines)
    print(table)

    # Plot one joint velocity (omega1) and angle (theta1) between nodes: velocity shows
    # the linear-vs-quadratic gap most clearly, since omega bends the most.
    problem, result = solve(PLOT_N)
    rc = perform_reality_check(problem, result, num_eval_points=400)
    t_fine = np.linspace(0.0, problem.duration, 800)
    x_quad, _ = trapezoidal_interpolate(problem.arm, result.time, result.state, result.control, t_fine)
    x_lin = np.column_stack([np.interp(t_fine, result.time, result.state[:, i]) for i in range(4)])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, i, name in [(axes[0], 0, r"$\theta_1$ [rad]"), (axes[1], 2, r"$\omega_1$ [rad/s]")]:
        ax.plot(rc.t_sim, rc.x_sim[:, i], color="black", lw=2.5, alpha=0.35, label="simulation (DOP853)")
        ax.plot(t_fine, x_lin[:, i], "--", color="tab:red", lw=1.3, label="linear (old)")
        ax.plot(t_fine, x_quad[:, i], "-", color="tab:blue", lw=1.3, label="quadratic (A1)")
        ax.plot(result.time, result.state[:, i], "o", color="tab:blue", ms=5, label="nodes")
        ax.set_xlabel("time [s]")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle(f"Trapezoidal state between nodes ({SCENARIO}, N = {PLOT_N})")
    fig.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "interpolation.png")
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
