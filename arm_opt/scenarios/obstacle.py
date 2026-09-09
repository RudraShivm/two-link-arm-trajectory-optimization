"""Scenario 3: Cartesian end-effector obstacle avoidance."""

from typing import Tuple
import numpy as np
from arm_opt.dynamics.parameters import ArmParameters
from arm_opt.dynamics.kinematics import forward_kinematics
from arm_opt.solvers.base import TrajectoryProblem
from arm_opt.scenarios.base_scenario import BaseScenario


class ObstacleScenario(BaseScenario):
    """Maneuver requiring the end-effector to steer around a circular workspace obstacle."""

    def __init__(
        self,
        obstacle_center: Tuple[float, float] = (1.2, 0.0),
        obstacle_radius: float = 0.35,
    ):
        self.obs_center = np.array(obstacle_center, dtype=float)
        self.obs_radius = float(obstacle_radius)

    @property
    def name(self) -> str:
        return "Cartesian Obstacle Avoidance"

    @property
    def description(self) -> str:
        return (
            f"End-effector must move from lower quadrant [-pi/4, 0] to upper quadrant [pi/4, 0] "
            f"while avoiding a circular keep-out obstacle at ({self.obs_center[0]:.2f}, {self.obs_center[1]:.2f}) "
            f"with radius {self.obs_radius:.2f}m. "
            "Enforced as a nonlinear state inequality constraint g(x) = ||p_ee(q) - p_obs||^2 - r^2 >= 0."
        )

    def create_problem(self, n_nodes: int = 30) -> TrajectoryProblem:
        x0 = np.array([-np.pi / 4, 0.0, 0.0, 0.0])
        xf = np.array([np.pi / 4, 0.0, 0.0, 0.0])
        params = ArmParameters(tau_max=40.0)

        # Obstacle avoidance inequality constraint: g(x, u) >= 0
        def obstacle_clearance(x: np.ndarray, u: np.ndarray) -> float:
            q = x[:2]
            _, p_ee = forward_kinematics(q, params)
            dist_sq = np.sum((p_ee - self.obs_center) ** 2)
            # Must stay outside radius plus slight safety margin (5mm)
            return float(dist_sq - (self.obs_radius + 0.005) ** 2)

        return TrajectoryProblem(
            x0=x0,
            xf=xf,
            duration=1.2,
            n_nodes=n_nodes,
            arm_params=params,
            tau_max=40.0,
            path_constraints=[obstacle_clearance],
        )

