"""Unit tests validating convergence and accuracy of the three trajectory optimization solvers."""

import unittest
import numpy as np
from arm_opt.dynamics.parameters import ArmParameters
from arm_opt.solvers.base import TrajectoryProblem
from arm_opt.solvers.shooting import ShootingSolver
from arm_opt.solvers.trapezoidal import TrapezoidalCollocationSolver
from arm_opt.solvers.hermite_simpson import HermiteSimpsonCollocationSolver


class TestSolvers(unittest.TestCase):
    def setUp(self):
        # Lightweight arm parameters for quick unit test execution
        self.params = ArmParameters(l1=1.0, l2=1.0, m1=1.0, m2=1.0, tau_max=50.0)
        self.x0 = np.array([0.0, 0.0, 0.0, 0.0])
        self.xf = np.array([np.pi / 4, -np.pi / 4, 0.0, 0.0])
        self.problem = TrajectoryProblem(
            x0=self.x0,
            xf=self.xf,
            duration=0.8,
            n_nodes=15,
            arm_params=self.params,
            tau_max=50.0,
        )

    def test_shooting_solver_convergence(self):
        solver = ShootingSolver(self.problem)
        result = solver.solve(max_iter=150, ftol=1e-5)
        self.assertTrue(result.success, f"Shooting failed: {result.message}")
        self.assertLess(result.max_constraint_violation, 1e-2)
        # Check start and terminal state match
        np.testing.assert_allclose(result.state[0], self.x0, atol=1e-5)
        np.testing.assert_allclose(result.state[-1], self.xf, atol=1e-2)

    def test_trapezoidal_solver_convergence(self):
        solver = TrapezoidalCollocationSolver(self.problem)
        result = solver.solve(max_iter=150, ftol=1e-5)
        self.assertTrue(result.success, f"Trapezoidal failed: {result.message}")
        self.assertLess(result.max_constraint_violation, 1e-3)
        np.testing.assert_allclose(result.state[0], self.x0, atol=1e-3)
        np.testing.assert_allclose(result.state[-1], self.xf, atol=1e-3)

    def test_hermite_simpson_solver_convergence(self):
        solver = HermiteSimpsonCollocationSolver(self.problem)
        result = solver.solve(max_iter=150, ftol=1e-5)
        self.assertTrue(result.success, f"Hermite-Simpson failed: {result.message}")
        self.assertLess(result.max_constraint_violation, 1e-3)
        np.testing.assert_allclose(result.state[0], self.x0, atol=1e-3)
        np.testing.assert_allclose(result.state[-1], self.xf, atol=1e-3)


if __name__ == "__main__":
    unittest.main()

