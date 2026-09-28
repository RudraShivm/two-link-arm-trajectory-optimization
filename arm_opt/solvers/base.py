from dataclasses import dataclass, field
from typing import List, Callable, Dict, Any
import numpy as np
from arm_opt.dynamics.parameters import ArmParameters, DEFAULT_PARAMS
from arm_opt.dynamics.manipulator import TwoLinkArm


@dataclass
class TrajectoryResult:
    method_name: str
    success: bool
    message: str
    time: np.ndarray
    state: np.ndarray
    control: np.ndarray
    cost: float
    solve_time: float
    iterations: int
    max_constraint_violation: float
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryProblem:
    """Fixed-horizon OCP: min int (||u||^2 + w||dq||^2) dt s.t. dynamics and bounds."""

    x0: np.ndarray
    xf: np.ndarray
    duration: float = 1.0
    n_nodes: int = 30
    arm_params: ArmParameters = DEFAULT_PARAMS
    tau_max: float = 25.0
    q_min: float = -2.0 * np.pi
    q_max: float = 2.0 * np.pi
    dq_max: float = 15.0
    effort_weight_dq: float = 0.001
    path_constraints: List[Callable[[np.ndarray, np.ndarray], float]] = field(
        default_factory=list
    )

    def __post_init__(self):
        self.arm = TwoLinkArm(self.arm_params)
        self.x0 = np.asarray(self.x0, dtype=float)
        self.xf = np.asarray(self.xf, dtype=float)

    @property
    def dt(self) -> float:
        return self.duration / self.n_nodes

    @property
    def time_grid(self) -> np.ndarray:
        return np.linspace(0.0, self.duration, self.n_nodes + 1)

    def get_linear_state_guess(self) -> np.ndarray:
        return np.linspace(self.x0, self.xf, self.n_nodes + 1)
