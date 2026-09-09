"""Publication-quality plotting functions for trajectory optimization comparisons."""

from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from arm_opt.dynamics.kinematics import forward_kinematics
from arm_opt.solvers.base import TrajectoryProblem, TrajectoryResult


# Consistent color styling across methods
METHOD_COLORS = {
    "Single Shooting (RK4)": "#D9381E",  # Red
    "Trapezoidal Collocation": "#1E6FD9",  # Blue
    "Hermite-Simpson Collocation": "#1E9E4A",  # Green
}


def plot_comparative_summary(
    problem: TrajectoryProblem,
    results: Dict[str, TrajectoryResult],
    obstacle_spec: Optional[Tuple[Tuple[float, float], float]] = None,
    save_path: Optional[str] = None,
):
    """Generates a 4-panel comparison figure showing Workspace, Angles, Velocities, and Torques."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax_ws = axes[0, 0]  # Cartesian workspace
    ax_q = axes[0, 1]  # Joint angles
    ax_dq = axes[1, 0]  # Joint velocities
    ax_tau = axes[1, 1]  # Actuator torques

    # 1. Cartesian Workspace
    if obstacle_spec is not None:
        (cx, cy), radius = obstacle_spec
        circle = plt.Circle(
            (cx, cy),
            radius,
            color="#D9534F",
            facecolor="#D9534F",
            alpha=0.35,
            edgecolor="#A94442",
            linestyle="--",
            label="Obstacle",
        )
        ax_ws.add_patch(circle)

    # Plot initial and target poses
    p_params = problem.arm_params
    elb_0, ee_0 = forward_kinematics(problem.x0[:2], p_params)
    elb_f, ee_f = forward_kinematics(problem.xf[:2], p_params)

    ax_ws.plot([0, elb_0[0], ee_0[0]], [0, elb_0[1], ee_0[1]], "k--", alpha=0.4, label="Initial Pose")
    ax_ws.plot([0, elb_f[0], ee_f[0]], [0, elb_f[1], ee_f[1]], "k-.", alpha=0.4, label="Target Pose")
    ax_ws.scatter([ee_f[0]], [ee_f[1]], color="#FFB300", marker="*", s=160, zorder=5, label="Target EE")

    for key, res in results.items():
        color = METHOD_COLORS.get(res.method_name, "#555555")
        t = res.time
        q = res.state[:, :2]
        dq = res.state[:, 2:]
        tau = res.control

        # Cartesian EE path
        ee_path = np.array([forward_kinematics(q_k, p_params)[1] for q_k in q])
        ax_ws.plot(ee_path[:, 0], ee_path[:, 1], color=color, linewidth=2.2, label=res.method_name)

        # Joint angles
        ax_q.plot(t, q[:, 0], color=color, linestyle="-", label=f"{res.method_name} - q1")
        ax_q.plot(t, q[:, 1], color=color, linestyle="--", alpha=0.75, label=f"{res.method_name} - q2")

        # Joint velocities
        ax_dq.plot(t, dq[:, 0], color=color, linestyle="-", label=f"{res.method_name} - dq1")
        ax_dq.plot(t, dq[:, 1], color=color, linestyle="--", alpha=0.75, label=f"{res.method_name} - dq2")

        # Torques
        ax_tau.plot(t, tau[:, 0], color=color, linestyle="-", label=f"{res.method_name} - tau1")
        ax_tau.plot(t, tau[:, 1], color=color, linestyle="--", alpha=0.75, label=f"{res.method_name} - tau2")

    # Torque limit lines
    ax_tau.axhline(problem.tau_max, color="red", linestyle=":", alpha=0.6, label="Torque Bound")
    ax_tau.axhline(-problem.tau_max, color="red", linestyle=":", alpha=0.6)

    ax_ws.set_title("Cartesian End-Effector Trajectories", fontsize=12, fontweight="bold")
    ax_ws.set_xlabel("X Position (m)")
    ax_ws.set_ylabel("Y Position (m)")
    ax_ws.axis("equal")
    ax_ws.legend(loc="best", fontsize=8)

    ax_q.set_title("Joint Angles q(t)", fontsize=12, fontweight="bold")
    ax_q.set_xlabel("Time (s)")
    ax_q.set_ylabel("Angle (rad)")
    ax_q.legend(loc="best", fontsize=8)

    ax_dq.set_title("Joint Velocities dq/dt", fontsize=12, fontweight="bold")
    ax_dq.set_xlabel("Time (s)")
    ax_dq.set_ylabel("Velocity (rad/s)")
    ax_dq.legend(loc="best", fontsize=8)

    ax_tau.set_title("Actuator Control Torques tau(t)", fontsize=12, fontweight="bold")
    ax_tau.set_xlabel("Time (s)")
    ax_tau.set_ylabel("Torque (N*m)")
    ax_tau.legend(loc="best", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

