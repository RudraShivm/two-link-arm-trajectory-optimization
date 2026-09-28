from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from arm_opt.dynamics.kinematics import forward_kinematics
from arm_opt.solvers.base import TrajectoryProblem, TrajectoryResult


METHOD_COLORS = {
    "Single Shooting (RK4)": "#D9381E",
    "Trapezoidal Collocation": "#1E6FD9",
    "Hermite-Simpson Collocation": "#1E9E4A",
}


def plot_comparative_summary(
    problem: TrajectoryProblem,
    results: Dict[str, TrajectoryResult],
    obstacle_spec: Optional[Tuple[Tuple[float, float], float]] = None,
    save_path: Optional[str] = None,
):
    style = (
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    plt.style.use(style)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_ws, ax_q, ax_dq, ax_tau = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    if obstacle_spec is not None:
        (cx, cy), radius = obstacle_spec
        ax_ws.add_patch(
            plt.Circle(
                (cx, cy),
                radius,
                facecolor="#D9534F",
                alpha=0.35,
                edgecolor="#A94442",
                linestyle="--",
                label="Obstacle",
            )
        )

    p_params = problem.arm_params
    elb_0, ee_0 = forward_kinematics(problem.x0[:2], p_params)
    elb_f, ee_f = forward_kinematics(problem.xf[:2], p_params)

    ax_ws.plot([0, elb_0[0], ee_0[0]], [0, elb_0[1], ee_0[1]], "k--", alpha=0.4, label="Initial")
    ax_ws.plot([0, elb_f[0], ee_f[0]], [0, elb_f[1], ee_f[1]], "k-.", alpha=0.4, label="Target")
    ax_ws.scatter([ee_f[0]], [ee_f[1]], color="#FFB300", marker="*", s=160, zorder=5, label="Target EE")

    for res in results.values():
        color = METHOD_COLORS.get(res.method_name, "#555555")
        t = res.time
        q = res.state[:, :2]
        dq = res.state[:, 2:]
        tau = res.control

        ee_path = np.array([forward_kinematics(q_k, p_params)[1] for q_k in q])
        ax_ws.plot(ee_path[:, 0], ee_path[:, 1], color=color, linewidth=2.2, label=res.method_name)
        ax_q.plot(t, q[:, 0], color=color, linestyle="-", label=f"{res.method_name} q1")
        ax_q.plot(t, q[:, 1], color=color, linestyle="--", alpha=0.75, label=f"{res.method_name} q2")
        ax_dq.plot(t, dq[:, 0], color=color, linestyle="-", label=f"{res.method_name} dq1")
        ax_dq.plot(t, dq[:, 1], color=color, linestyle="--", alpha=0.75, label=f"{res.method_name} dq2")
        ax_tau.plot(t, tau[:, 0], color=color, linestyle="-", label=f"{res.method_name} tau1")
        ax_tau.plot(t, tau[:, 1], color=color, linestyle="--", alpha=0.75, label=f"{res.method_name} tau2")

    ax_tau.axhline(problem.tau_max, color="red", linestyle=":", alpha=0.6, label="Torque bound")
    ax_tau.axhline(-problem.tau_max, color="red", linestyle=":", alpha=0.6)

    ax_ws.set_title("End-effector paths")
    ax_ws.set_xlabel("X (m)")
    ax_ws.set_ylabel("Y (m)")
    ax_ws.axis("equal")
    ax_ws.legend(loc="best", fontsize=8)

    ax_q.set_title("Joint angles")
    ax_q.set_xlabel("Time (s)")
    ax_q.set_ylabel("rad")
    ax_q.legend(loc="best", fontsize=8)

    ax_dq.set_title("Joint velocities")
    ax_dq.set_xlabel("Time (s)")
    ax_dq.set_ylabel("rad/s")
    ax_dq.legend(loc="best", fontsize=8)

    ax_tau.set_title("Torques")
    ax_tau.set_xlabel("Time (s)")
    ax_tau.set_ylabel("N*m")
    ax_tau.legend(loc="best", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
