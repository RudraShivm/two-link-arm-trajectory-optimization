"""Scenario 1: Rest-to-rest maneuver."""

import numpy as np
from arm_opt.dynamics.parameters import ArmParameters
from arm_opt.solvers.base import TrajectoryProblem
from arm_opt.scenarios.base_scenario import BaseScenario


class RestToRestScenario(BaseScenario):
    """Rest-to-rest motion from horizontal extension to upright vertical."""

    @property
    def name(self) -> str:
        return "Rest-to-Rest Baseline"

    @property
    def description(self) -> str:
        return (
            "Standard sanity benchmark moving from horizontal extension [0, 0] "
            "to upright vertical [pi/2, 0] in 1.0s under moderate torque limits (30 N*m). "
            "All three numerical algorithms should converge cleanly."
        )

    def create_problem(self, n_nodes: int = 30) -> TrajectoryProblem:
        x0 = np.array([0.0, 0.0, 0.0, 0.0])
        xf = np.array([np.pi / 2, 0.0, 0.0, 0.0])
        params = ArmParameters(tau_max=30.0)
        return TrajectoryProblem(
            x0=x0,
            xf=xf,
            duration=1.0,
            n_nodes=n_nodes,
            arm_params=params,
            tau_max=30.0,
        )

