import numpy as np
from arm_opt.dynamics.parameters import ArmParameters
from arm_opt.solvers.base import TrajectoryProblem
from arm_opt.scenarios.base_scenario import BaseScenario


class SwingUpScenario(BaseScenario):
    @property
    def name(self) -> str:
        return "Under-Torqued Swing-Up"

    @property
    def description(self) -> str:
        return (
            "Invert from hanging [-pi/2, 0] to upright [pi/2, 0] in 2.0 s. "
            "tau_max = 6 N*m is below peak static gravity load (~19.6 N*m), "
            "so the arm must pump energy through oscillation."
        )

    def create_problem(self, n_nodes: int = 40) -> TrajectoryProblem:
        x0 = np.array([-np.pi / 2, 0.0, 0.0, 0.0])
        xf = np.array([np.pi / 2, 0.0, 0.0, 0.0])
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
