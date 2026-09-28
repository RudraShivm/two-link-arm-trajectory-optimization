"""Scenario 3: Cartesian end-effector obstacle avoidance."""

from typing import Tuple
import numpy as np
from arm_opt.dynamics.parameters import ArmParameters
from arm_opt.dynamics.kinematics import forward_kinematics
from arm_opt.solvers.base import TrajectoryProblem
from arm_opt.scenarios.base_scenario import BaseScenario


def segment_clearance_sq(a: np.ndarray, b: np.ndarray, center: np.ndarray, radius: float) -> float:
    """Squared distance from `center` to the segment a-b, minus radius^2 (>= 0 means outside).

    Uses the exact closest point on the segment (projection clamped to [0, 1]),
    so the circle cannot slip between sample points.
    """
    ab = b - a
    s = np.clip(np.dot(center - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    closest = a + s * ab
    return float(np.sum((closest - center) ** 2) - radius**2)


class ObstacleScenario(BaseScenario):
    """Maneuver requiring the end-effector to steer around a circular workspace obstacle."""

    # Obstacle for the whole-arm check (A3). The default (1.2, 0) obstacle cannot be
    # used with it: at theta1 = 0 link 1 lies on [0, 1] x {0}, only 0.2 m from its center.
    # This one is out of link 1's reach (|c| - r = 1.23 m > l1) but inside link 2's sweep.
    FULL_BODY_CENTER = (1.5, -0.3)
    FULL_BODY_RADIUS = 0.3

    def __init__(
        self,
        obstacle_center: Tuple[float, float] = (1.2, 0.0),
        obstacle_radius: float = 0.35,
        check_full_body: bool = False,
    ):
        """
        Args:
            check_full_body: If True, both links (not just the end-effector) must stay
                outside the obstacle. Default False keeps the original hand-only check.
        """
        self.obs_center = np.array(obstacle_center, dtype=float)
        self.obs_radius = float(obstacle_radius)
        self.check_full_body = check_full_body

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
            + (" Whole-arm check: both links must also stay outside." if self.check_full_body else "")
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

        path_constraints = [obstacle_clearance]
        if self.check_full_body:
            r_safe = self.obs_radius + 0.005

            def link1_clearance(x: np.ndarray, u: np.ndarray) -> float:
                p_elbow, _ = forward_kinematics(x[:2], params)
                return segment_clearance_sq(np.zeros(2), p_elbow, self.obs_center, r_safe)

            def link2_clearance(x: np.ndarray, u: np.ndarray) -> float:
                p_elbow, p_ee = forward_kinematics(x[:2], params)
                return segment_clearance_sq(p_elbow, p_ee, self.obs_center, r_safe)

            # Link 2 ends at the hand, so it also covers the end-effector check.
            path_constraints = [link1_clearance, link2_clearance]

        return TrajectoryProblem(
            x0=x0,
            xf=xf,
            duration=1.2,
            n_nodes=n_nodes,
            arm_params=params,
            tau_max=40.0,
            path_constraints=path_constraints,
        )

