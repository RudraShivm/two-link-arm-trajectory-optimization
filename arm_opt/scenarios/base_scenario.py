"""Base class and common definitions for benchmark scenarios."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from arm_opt.solvers.base import TrajectoryProblem, TrajectoryResult
from arm_opt.solvers.shooting import ShootingSolver
from arm_opt.solvers.trapezoidal import TrapezoidalCollocationSolver
from arm_opt.solvers.hermite_simpson import HermiteSimpsonCollocationSolver


class BaseScenario(ABC):
    """Abstract base class defining a trajectory optimization benchmark challenge."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable scenario name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Detailed scenario description explaining physical challenges."""
        pass

    @abstractmethod
    def create_problem(self, n_nodes: int = 30) -> TrajectoryProblem:
        """Constructs the TrajectoryProblem instance with initial/terminal conditions."""
        pass

    def run_comparison(
        self,
        n_nodes: int = 30,
        solvers: Optional[List[str]] = None,
        max_iter: int = 400,
        ftol: float = 1e-5,
    ) -> Dict[str, TrajectoryResult]:
        """Runs all specified solvers on this scenario and collects results."""
        if solvers is None:
            solvers = ["shooting", "trapezoidal", "hermite_simpson"]

        problem = self.create_problem(n_nodes=n_nodes)
        results = {}

        if "shooting" in solvers:
            s_solver = ShootingSolver(problem)
            results["shooting"] = s_solver.solve(max_iter=max_iter, ftol=ftol)

        if "trapezoidal" in solvers:
            t_solver = TrapezoidalCollocationSolver(problem)
            results["trapezoidal"] = t_solver.solve(max_iter=max_iter, ftol=ftol)

        if "hermite_simpson" in solvers:
            h_solver = HermiteSimpsonCollocationSolver(problem)
            results["hermite_simpson"] = h_solver.solve(max_iter=max_iter, ftol=ftol)

        return results

