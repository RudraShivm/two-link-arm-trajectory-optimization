import numpy as np
from arm_opt.dynamics.parameters import ArmParameters
from arm_opt.solvers.base import TrajectoryProblem
from arm_opt.scenarios.base_scenario import BaseScenario


class RestToRestScenario(BaseScenario):
    @property
    def name(self) -> str:
        return "Rest-to-Rest Baseline"

    @property
    def description(self) -> str:
        return (
            "Move from horizontal [0, 0] to upright [pi/2, 0] in 1.0 s "
            "with tau_max = 30 N*m."
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
