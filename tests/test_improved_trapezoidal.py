"""Unit tests for the improved trapezoidal solver (S1-S5).

Two kinds of tests:
  1. Building blocks are correct (batch physics, gradient, sparse Jacobian).
  2. Every improvement gives the SAME answer as the original solver.

Run:  PYTHONPATH=. python3 -m unittest tests.test_improved_trapezoidal -v
"""

import unittest

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from arm_opt.dynamics.manipulator import TwoLinkArm
from arm_opt.dynamics.parameters import ArmParameters
from arm_opt.scenarios import SCENARIOS
from arm_opt.solvers.base import TrajectoryProblem
from arm_opt.solvers.improved_trapezoidal import (
    ImprovedTrapezoidalSolver,
    build_pattern,
    color_columns,
    default_coarse_levels,
    solve_with_warm_start,
    sparse_fd_jacobian,
    state_derivative_batch,
)
from arm_opt.solvers.trapezoidal import TrapezoidalCollocationSolver

ALL_OFF = dict(fix_endpoints_in_bounds=False, vectorized=False, sparse_jacobian=False, scaling=False)


def small_problem():
    """Same small problem as tests/test_solvers.py (fast to solve)."""
    params = ArmParameters(l1=1.0, l2=1.0, m1=1.0, m2=1.0, tau_max=50.0)
    return TrajectoryProblem(
        x0=np.array([0.0, 0.0, 0.0, 0.0]),
        xf=np.array([np.pi / 4, -np.pi / 4, 0.0, 0.0]),
        duration=0.8,
        n_nodes=15,
        arm_params=params,
        tau_max=50.0,
    )


class TestBuildingBlocks(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.problem = small_problem()

    def test_batch_dynamics_matches_loop(self):
        """S2: vectorized f(x, u) equals the original one-node-at-a-time version."""
        arm = TwoLinkArm(self.problem.arm_params)
        X = self.rng.normal(size=(1000, 4)) * 3.0
        U = self.rng.normal(size=(1000, 2)) * 20.0
        loop = np.array([arm.state_derivative(x, u) for x, u in zip(X, U)])
        batch = state_derivative_batch(self.problem.arm_params, X, U)
        np.testing.assert_allclose(batch, loop, rtol=1e-10, atol=1e-10)

    def test_objective_gradient_matches_finite_difference(self):
        """S3(a): exact objective gradient equals a numerical estimate."""
        solver = ImprovedTrapezoidalSolver(self.problem)
        z = self.rng.normal(size=solver.n_vars)
        numeric = approx_derivative(solver.objective, z)
        np.testing.assert_allclose(solver.objective_grad(z), numeric, atol=1e-5)

    def test_defect_block_uses_12_color_groups(self):
        """S3(b): 6 variables x {even, odd nodes} = 12 evaluations instead of 6*(N+1)."""
        solver = ImprovedTrapezoidalSolver(self.problem)
        defect_block = [b for b in solver.constraint_blocks() if b["fun"] == solver.defects][0]
        S = build_pattern(defect_block["row_nodes"], solver.N + 1)
        self.assertEqual(len(color_columns(S)), 12)

    def test_sparse_jacobian_matches_dense(self):
        """S3(b): colored Jacobian equals a normal (dense) finite-difference Jacobian,
        for every constraint block, including start/end and obstacle blocks."""
        obstacle = SCENARIOS["obstacle"]().create_problem(n_nodes=10)
        for problem in (self.problem, obstacle):
            solver = ImprovedTrapezoidalSolver(problem, fix_endpoints_in_bounds=False)
            z = self.rng.normal(size=solver.n_vars)
            for blk in solver.constraint_blocks():
                S = build_pattern(blk["row_nodes"], solver.N + 1)
                J_sparse = sparse_fd_jacobian(blk["fun"], z, S, color_columns(S))
                J_dense = approx_derivative(blk["fun"], z)
                np.testing.assert_allclose(J_sparse, J_dense, atol=1e-5)

    def test_bounds_lock_endpoints(self):
        """S1: min = max = required value for the first and last node's states."""
        solver = ImprovedTrapezoidalSolver(self.problem)
        lo, hi = solver._bounds()
        lo, hi = lo.reshape(-1, 6), hi.reshape(-1, 6)
        np.testing.assert_array_equal(lo[0, :4], self.problem.x0)
        np.testing.assert_array_equal(hi[0, :4], self.problem.x0)
        np.testing.assert_array_equal(lo[-1, :4], self.problem.xf)
        np.testing.assert_array_equal(hi[-1, :4], self.problem.xf)

    def test_default_coarse_levels(self):
        """S5: coarse grids are N/2, N/4, ... down to >= 8."""
        self.assertEqual(default_coarse_levels(30), [15])
        self.assertEqual(default_coarse_levels(40), [10, 20])
        self.assertEqual(default_coarse_levels(80), [10, 20, 40])
        self.assertEqual(default_coarse_levels(10), [])


class TestSameAnswerAsOriginal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = small_problem()
        cls.original = TrapezoidalCollocationSolver(cls.problem).solve(max_iter=150, ftol=1e-5)

    def check_same_answer(self, res):
        self.assertTrue(res.success, res.message)
        self.assertLess(res.max_constraint_violation, 1e-3)
        # Relative cost difference to the original solver
        self.assertLess(abs(res.cost - self.original.cost) / self.original.cost, 1e-3)
        np.testing.assert_allclose(res.state[0], self.problem.x0, atol=1e-3)
        np.testing.assert_allclose(res.state[-1], self.problem.xf, atol=1e-3)

    def test_all_flags_off_reproduces_original(self):
        res = ImprovedTrapezoidalSolver(self.problem, **ALL_OFF).solve(max_iter=150, ftol=1e-5)
        self.check_same_answer(res)

    def test_each_improvement_alone(self):
        for flag in ALL_OFF:
            with self.subTest(flag=flag):
                res = ImprovedTrapezoidalSolver(self.problem, **{**ALL_OFF, flag: True}).solve(
                    max_iter=150, ftol=1e-5)
                self.check_same_answer(res)

    def test_all_improvements(self):
        res = ImprovedTrapezoidalSolver(self.problem).solve(max_iter=150, ftol=1e-5)
        self.check_same_answer(res)

    def test_warm_start(self):
        """S5: same answer, and the reported time includes the coarse solves."""
        res = solve_with_warm_start(self.problem, coarse_levels=[8], max_iter=150, ftol=1e-5)
        self.check_same_answer(res)
        levels = res.info["levels"]
        self.assertEqual([lv["N"] for lv in levels], [8, 15])
        self.assertAlmostEqual(res.solve_time, sum(lv["time"] for lv in levels))
        self.assertEqual(res.iterations, sum(lv["iterations"] for lv in levels))


if __name__ == "__main__":
    unittest.main()
