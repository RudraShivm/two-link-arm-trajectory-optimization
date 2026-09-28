from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from arm_opt.solvers.base import TrajectoryProblem, TrajectoryResult
from arm_opt.solvers.shooting import ShootingSolver
from arm_opt.solvers.trapezoidal import TrapezoidalCollocationSolver
from arm_opt.solvers.hermite_simpson import HermiteSimpsonCollocationSolver


class BaseScenario(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def create_problem(self, n_nodes: int = 30) -> TrajectoryProblem:
        pass

    def run_comparison(
        self,
        n_nodes: int = 30,
        solvers: Optional[List[str]] = None,
        max_iter: int = 400,
        ftol: float = 1e-5,
    ) -> Dict[str, TrajectoryResult]:
        if solvers is None:
            solvers = ["shooting", "trapezoidal", "hermite_simpson"]

        problem = self.create_problem(n_nodes=n_nodes)
        results = {}

        if "shooting" in solvers:
            results["shooting"] = ShootingSolver(problem).solve(
                max_iter=max_iter, ftol=ftol
            )

        if "trapezoidal" in solvers:
            results["trapezoidal"] = TrapezoidalCollocationSolver(problem).solve(
                max_iter=max_iter, ftol=ftol
            )

        if "hermite_simpson" in solvers:
            results["hermite_simpson"] = HermiteSimpsonCollocationSolver(problem).solve(
                max_iter=max_iter, ftol=ftol
            )

        return results
