"""Scenario 6: 180-degree full workspace inversion across singularity."""

import numpy as np
from arm_opt.dynamics.parameters import ArmParameters
from arm_opt.solvers.base import TrajectoryProblem
from arm_opt.scenarios.base_scenario import BaseScenario


class ReversalScenario(BaseScenario):
    """Full workspace 180-degree horizontal inversion from extended [0, 0] to opposite [-pi, 0]."""

    @property
    def name(self) -> str:
        return "180-Degree Inversion Across Singularity"

    @property
    def description(self) -> str:
        return (
            "Demands a complete 180-degree planar inversion from fully outstretched horizontal [0, 0] "
            "(EE at [2.00, 0.00]m) to fully inverted horizontal [pi, 0] (EE at [-2.00, 0.00]m) in 1.0s. "
            "Traversing across the workspace boundary passes near kinematic singularities where the "
            "manipulator Jacobian conditioning deteriorates. "
            "Single Shooting and Trapezoidal Collocation diverge or fail to converge; only 4th-order "
            "Hermite-Simpson Collocation successfully resolves the nonlinear transition."
        )

    def create_problem(self, n_nodes: int = 20) -> TrajectoryProblem:
        x0 = np.array([0.0, 0.0, 0.0, 0.0])
        xf = np.array([np.pi, 0.0, 0.0, 0.0])
        tau_limit = 45.0
        params = ArmParameters(tau_max=tau_limit)
        return TrajectoryProblem(
            x0=x0,
            xf=xf,
            duration=1.0,
            n_nodes=n_nodes,
            arm_params=params,
            tau_max=tau_limit,
            effort_weight_dq=0.0001,
        )