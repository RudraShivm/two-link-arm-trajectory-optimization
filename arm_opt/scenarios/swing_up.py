"""Scenario 2: Under-torqued double pendulum swing-up (Acrobot challenge)."""

import numpy as np
from arm_opt.dynamics.parameters import ArmParameters
from arm_opt.solvers.base import TrajectoryProblem
from arm_opt.scenarios.base_scenario import BaseScenario


class SwingUpScenario(BaseScenario):
    """Under-torqued double-pendulum inversion from hanging down to upright."""

    @property
    def name(self) -> str:
        return "Under-Torqued Swing-Up (Acrobot)"

    @property
    def description(self) -> str:
        return (
            "Starts hanging vertically downwards [-pi/2, 0] and must reach inverted upright [pi/2, 0]. "
            "Actuator torque is strictly limited to 6.0 N*m, far below the peak gravitational stall torque "
            "(~14.7 N*m). The arm cannot raise itself directly; it must pump energy back and forth. "
            "Single Shooting typically diverges due to chaotic sensitivity over the 2.0s horizon, "
            "demonstrating the structural superiority of Direct Collocation."
        )

    def create_problem(self, n_nodes: int = 40) -> TrajectoryProblem:
        x0 = np.array([-np.pi / 2, 0.0, 0.0, 0.0])
        xf = np.array([np.pi / 2, 0.0, 0.0, 0.0])
        # Peak static gravity torque is ~14.7 N*m; limit to 6.0 N*m
        tau_limit = 6.0
        params = ArmParameters(tau_max=tau_limit)
        return TrajectoryProblem(
            x0=x0,
            xf=xf,
            duration=2.0,
            n_nodes=n_nodes,
            arm_params=params,
            tau_max=tau_limit,
            effort_weight_dq=0.0001,
        )

