from typing import Tuple
import numpy as np
from arm_opt.dynamics.parameters import ArmParameters
from arm_opt.dynamics.kinematics import forward_kinematics
from arm_opt.solvers.base import TrajectoryProblem
from arm_opt.scenarios.base_scenario import BaseScenario


class ObstacleScenario(BaseScenario):
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
            f"Move from [-pi/4, 0] to [pi/4, 0] while keeping the end-effector "
            f"outside a circle at ({self.obs_center[0]:.2f}, {self.obs_center[1]:.2f}) "
            f"with radius {self.obs_radius:.2f} m."
        )

    def create_problem(self, n_nodes: int = 30) -> TrajectoryProblem:
        x0 = np.array([-np.pi / 4, 0.0, 0.0, 0.0])
        xf = np.array([np.pi / 4, 0.0, 0.0, 0.0])
        params = ArmParameters(tau_max=40.0)

        def obstacle_clearance(x: np.ndarray, u: np.ndarray) -> float:
            _, p_ee = forward_kinematics(x[:2], params)
            dist_sq = np.sum((p_ee - self.obs_center) ** 2)
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
