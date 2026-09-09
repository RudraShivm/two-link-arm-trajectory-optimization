"""Scenario 5: Under-torqued inversion requiring energy pumping."""

import numpy as np
from arm_opt.dynamics.parameters import ArmParameters
from arm_opt.solvers.base import TrajectoryProblem
from arm_opt.scenarios.base_scenario import BaseScenario


class UnderTorquedScenario(BaseScenario):
    """Under-torqued arm inversion from downward hanging to upright vertical."""

    @property
    def name(self) -> str:
        return "Under-Torqued Inversion (Energy Pumping)"

    @property
    def description(self) -> str:
        return (
            "Starts hanging downward [-pi/2, 0] and must invert to upright vertical [pi/2, 0]. "
            "Actuator torque is strictly limited to 10.0 N*m, strictly below the static gravitational "
            "stall torque (~14.7 N*m). The manipulator cannot ascend directly; it must pump mechanical "
            "energy by executing a counter-intuitive initial backswing. "
            "Single Shooting diverges due to extreme initial-condition sensitivity, while Direct "
            "Collocation discovers the resonant pumping orbit."
        )

    def create_problem(self, n_nodes: int = 18) -> TrajectoryProblem:
        x0 = np.array([-np.pi / 2, 0.0, 0.0, 0.0])
        xf = np.array([np.pi / 2, 0.0, 0.0, 0.0])
        tau_limit = 10.0
        params = ArmParameters(tau_max=tau_limit)
        return TrajectoryProblem(
            x0=x0,
            xf=xf,
            duration=1.2,
            n_nodes=n_nodes,
            arm_params=params,
            tau_max=tau_limit,
            effort_weight_dq=0.0001,
        )