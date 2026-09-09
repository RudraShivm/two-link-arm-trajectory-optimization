"""Trajectory optimization solvers package."""

from arm_opt.solvers.base import TrajectoryProblem, TrajectoryResult
from arm_opt.solvers.shooting import ShootingSolver
from arm_opt.solvers.trapezoidal import TrapezoidalCollocationSolver
from arm_opt.solvers.hermite_simpson import HermiteSimpsonCollocationSolver

__all__ = [
    "TrajectoryProblem",
    "TrajectoryResult",
    "ShootingSolver",
    "TrapezoidalCollocationSolver",
    "HermiteSimpsonCollocationSolver",
]

