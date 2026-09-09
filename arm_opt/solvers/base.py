"""Base data structures and abstract problem interfaces for trajectory optimization."""

from dataclasses import dataclass, field
from typing import List, Callable, Optional, Dict, Any
import numpy as np
from arm_opt.dynamics.parameters import ArmParameters, DEFAULT_PARAMS
from arm_opt.dynamics.manipulator import TwoLinkArm


@dataclass
class TrajectoryResult:
    """Standardized output container for trajectory optimization solutions.

    Attributes:
        method_name: Name of the numerical method used.
        success: Whether the numerical optimizer converged successfully.
        message: Status message from the optimizer.
        time: Discrete time grid array of shape (K,).
        state: State trajectory [q, dq] of shape (K, 4).
        control: Control torque trajectory [tau] of shape (K, 2).
        cost: Evaluated objective cost value.
        solve_time: Wall-clock optimization time in seconds.
        iterations: Number of optimizer iterations.
        max_constraint_violation: Maximum absolute constraint violation.
        info: Additional solver-specific diagnostic data.
    """

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
    """Specification of a fixed-horizon optimal control problem.

    Minimizes:
        int_0^T [ ||tau(t)||^2 + effort_weight_dq * ||dq(t)||^2 ] dt
    Subject to:
        dx/dt = f(x, tau)
        x(0) = x0
        x(T) = xf
        -tau_max <= tau <= tau_max
        q_min <= q <= q_max
        -dq_max <= dq <= dq_max
        path_constraints(x, tau) >= 0  (optional)
    """

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
        """Time step between collocation grid nodes."""
        return self.duration / self.n_nodes

    @property
    def time_grid(self) -> np.ndarray:
        """Discrete time points from 0 to duration."""
        return np.linspace(0.0, self.duration, self.n_nodes + 1)

    def get_linear_state_guess(self) -> np.ndarray:
        """Generates a simple linear state interpolation between x0 and xf of shape (n_nodes+1, 4)."""
        return np.linspace(self.x0, self.xf, self.n_nodes + 1)

