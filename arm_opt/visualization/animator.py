"""Synchronized multi-panel video animation of robotic arm trajectories."""

from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import numpy as np
from scipy.interpolate import interp1d
from arm_opt.dynamics.kinematics import forward_kinematics
from arm_opt.solvers.base import TrajectoryProblem, TrajectoryResult


class SynchronizedArmAnimator:
    """Generates a synchronized side-by-side video animation comparing trajectory solutions."""

    def __init__(
        self,
        problem: TrajectoryProblem,
        results: Dict[str, TrajectoryResult],
        obstacle_spec: Optional[Tuple[Tuple[float, float], float]] = None,
        fps: int = 50,
    ):
        self.problem = problem
        self.results = results
        self.obstacle_spec = obstacle_spec
        self.fps = fps
        self.params = problem.arm_params

        # Prepare uniform continuous time grid for smooth animation
        duration = problem.duration
        self.num_frames = int(duration * fps) + 1
        self.t_eval = np.linspace(0.0, duration, self.num_frames)

        # Interpolate states, controls, and joint Cartesian positions for all methods
        self.interpolated = {}
        for key, res in results.items():
            t_orig = res.time
            q_interp = interp1d(t_orig, res.state[:, :2], axis=0, kind="linear")(self.t_eval)
            tau_interp = interp1d(t_orig, res.control, axis=0, kind="linear")(self.t_eval)

            # Compute Cartesian coordinates for each frame
            n_frames = len(self.t_eval)
            p_elbow = np.zeros((n_frames, 2))
            p_ee = np.zeros((n_frames, 2))
            for f in range(n_frames):
                p_elbow[f], p_ee[f] = forward_kinematics(q_interp[f], self.params)

            self.interpolated[key] = {
                "name": res.method_name,
                "q": q_interp,
                "tau": tau_interp,
                "p_elbow": p_elbow,
                "p_ee": p_ee,
            }

    def render_video(self, output_path: str):
        """Renders and saves the animation as MP4 (via ffmpeg) or GIF."""
        methods = list(self.interpolated.keys())
        n_methods = len(methods)

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, axes = plt.subplots(1, n_methods, figsize=(6.0 * n_methods, 6.5), squeeze=False)
        axes = axes.flatten()

        _, target_ee = forward_kinematics(self.problem.xf[:2], self.params)
        reach = self.params.l1 + self.params.l2 + 0.35

        lines_arm = []
        scatter_joints = []
        lines_trail = []
        text_time = []
        text_torques = []

        method_colors = {
            "shooting": "#D9381E",
            "trapezoidal": "#1E6FD9",
            "hermite_simpson": "#1E9E4A",
        }

        for i, m_key in enumerate(methods):
            ax = axes[i]
            data = self.interpolated[m_key]
            color = method_colors.get(m_key, "#333333")

            ax.set_xlim(-reach, reach)
            ax.set_ylim(-reach, reach)
            ax.set_aspect("equal")
            ax.set_title(data["name"], fontsize=13, fontweight="bold", pad=12)
            ax.set_xlabel("X Position (m)")
            if i == 0:
                ax.set_ylabel("Y Position (m)")

            # Obstacle if defined
            if self.obstacle_spec is not None:
                (cx, cy), radius = self.obstacle_spec
                obs_patch = plt.Circle(
                    (cx, cy),
                    radius,
                    color="#E74C3C",
                    facecolor="#E74C3C",
                    alpha=0.35,
                    edgecolor="#C0392B",
                    linewidth=2.0,
                    linestyle="--",
                )
                ax.add_patch(obs_patch)

            # Target marker
            ax.scatter(
                [target_ee[0]],
                [target_ee[1]],
                color="#F39C12",
                edgecolors="black",
                marker="*",
                s=240,
                zorder=6,
                label="Target",
            )

            # Arm link line
            (line,) = ax.plot([], [], "-", color=color, linewidth=5.5, solid_capstyle="round", zorder=4)
            lines_arm.append(line)

            # Joint spheres
            scat = ax.scatter([], [], color="#2C3E50", s=90, zorder=5)
            scatter_joints.append(scat)

            # End-effector trailing ribbon
            (trail,) = ax.plot([], [], "--", color=color, linewidth=1.6, alpha=0.6, zorder=3)
            lines_trail.append(trail)

            # Real-time info overlay
            t_txt = ax.text(
                0.04,
                0.94,
                "",
                transform=ax.transAxes,
                fontsize=10,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
            )
            text_time.append(t_txt)

            tau_txt = ax.text(
                0.04,
                0.04,
                "",
                transform=ax.transAxes,
                fontsize=9.5,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
            )
            text_torques.append(tau_txt)

        def init():
            for i in range(n_methods):
                lines_arm[i].set_data([], [])
                scatter_joints[i].set_offsets(np.empty((0, 2)))
                lines_trail[i].set_data([], [])
                text_time[i].set_text("")
                text_torques[i].set_text("")
            return lines_arm + scatter_joints + lines_trail + text_time + text_torques

        def update(frame):
            current_time = self.t_eval[frame]

            for i, m_key in enumerate(methods):
                data = self.interpolated[m_key]
                p_elb = data["p_elbow"][frame]
                p_ee = data["p_ee"][frame]
                tau = data["tau"][frame]

                # Update arm rods
                arm_x = [0.0, p_elb[0], p_ee[0]]
                arm_y = [0.0, p_elb[1], p_ee[1]]
                lines_arm[i].set_data(arm_x, arm_y)

                # Update joint circles
                scatter_joints[i].set_offsets([[0.0, 0.0], [p_elb[0], p_elb[1]], [p_ee[0], p_ee[1]]])

                # Update trailing ribbon
                start_trail = max(0, frame - int(self.fps * 0.75))
                trail_x = data["p_ee"][start_trail : frame + 1, 0]
                trail_y = data["p_ee"][start_trail : frame + 1, 1]
                lines_trail[i].set_data(trail_x, trail_y)

                # Update text indicators
                text_time[i].set_text(f"t = {current_time:.2f} s / {self.problem.duration:.2f} s")
                text_torques[i].set_text(
                    f"tau1: {tau[0]:+6.2f} N*m\n"
                    f"tau2: {tau[1]:+6.2f} N*m"
                )

            return lines_arm + scatter_joints + lines_trail + text_time + text_torques

        anim = FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=self.num_frames,
            interval=1000.0 / self.fps,
            blit=False,
        )

        plt.tight_layout()

        if output_path.endswith(".mp4"):
            writer = FFMpegWriter(fps=self.fps, metadata=dict(artist="arm_opt"), bitrate=2400)
            anim.save(output_path, writer=writer)
        elif output_path.endswith(".gif"):
            writer = PillowWriter(fps=min(self.fps, 25))
            anim.save(output_path, writer=writer)
        else:
            raise ValueError(f"Unsupported video format: {output_path}")

        plt.close()

