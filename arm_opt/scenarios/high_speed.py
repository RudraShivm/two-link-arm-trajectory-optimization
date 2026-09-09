"""Scenario 4: High-speed dynamic whip maneuver."""

import numpy as np
from arm_opt.dynamics.parameters import ArmParameters
from arm_opt.solvers.base import TrajectoryProblem
from arm_opt.scenarios.base_scenario import BaseScenario


class HighSpeedScenario(BaseScenario):
    """Aggressive high-velocity maneuver where nonlinear Coriolis forces dominate."""

    @property
    def name(self) -> str:
        return "High-Speed Dynamic Maneuver"

    @property
    def description(self) -> str:
        return (
            "Executes a large angular stroke from [-pi/3, pi/4] to [pi/2, -pi/3] "
            "in an aggressive 0.45s time budget. At high joint velocities, "
            "Coriolis and centrifugal terms (quadratic in velocity) dominate the dynamics. "
            "Low-order piecewise linear methods (Trapezoidal) suffer from severe discretization "
            "defects unless fine grids are used, while Hermite-Simpson retains high fidelity."
        )

    def create_problem(self, n_nodes: int = 25) -> TrajectoryProblem:
        x0 = np.array([-np.pi / 3, np.pi / 4, 0.0, 0.0])
        xf = np.array([np.pi / 2, -np.pi / 3, 0.0, 0.0])
        params = ArmParameters(tau_max=80.0, dq_max=25.0)
        return TrajectoryProblem(
            x0=x0,
            xf=xf,
            duration=0.45,
            n_nodes=n_nodes,
            arm_params=params,
            tau_max=80.0,
            dq_max=25.0,
            effort_weight_dq=0.0001,
        )

