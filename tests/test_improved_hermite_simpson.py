"""Unit tests for the improved Hermite-Simpson solver (S1, S2, S3, S5).

Two kinds of tests, same split as tests/test_improved_trapezoidal.py:
  1. Building blocks are correct (batch physics, gradient, sparse Jacobian).
  2. Every improvement gives the SAME answer as the original solver.

Run:  PYTHONPATH=. python3 -m unittest tests.test_improved_hermite_simpson -v
"""

import unittest

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from arm_opt.dynamics.parameters import ArmParameters
from arm_opt.scenarios import SCENARIOS
from arm_opt.solvers.base import TrajectoryProblem
from arm_opt.solvers.hermite_simpson import HermiteSimpsonCollocationSolver
from arm_opt.solvers.improved_hermite_simpson import (
    ImprovedHermiteSimpsonSolver,
    build_pattern,
    color_columns,
    default_coarse_levels,
    solve_with_warm_start,
    sparse_fd_jacobian,
    state_derivative_batch,
)

ALL_OFF = dict(fix_endpoints_in_bounds=False, vectorized=False, sparse_jacobian=False)


def small_problem():
    """Same small problem style as tests/test_improved_trapezoidal.py (fast to solve)."""
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
        """S2: vectorized f(x, u) equals the original one-point-at-a-time version.

        Reuses state_derivative_batch from improved_trapezoidal.py directly, so
        this is really re-confirming it also holds for Hermite-Simpson's usage
        pattern (mesh nodes AND midpoints mixed in the same array).
        """
        from arm_opt.dynamics.manipulator import TwoLinkArm

        arm = TwoLinkArm(self.problem.arm_params)
        X = self.rng.normal(size=(1000, 4)) * 3.0
        U = self.rng.normal(size=(1000, 2)) * 20.0
        loop = np.array([arm.state_derivative(x, u) for x, u in zip(X, U)])
        batch = state_derivative_batch(self.problem.arm_params, X, U)
        np.testing.assert_allclose(batch, loop, rtol=1e-10, atol=1e-10)

    def test_objective_matches_original_loop(self):
        """The vectorized Simpson-quadrature objective equals the original's
        loop-based version (copied inline here from hermite_simpson.py)."""
        solver = ImprovedHermiteSimpsonSolver(self.problem)
        z = self.rng.normal(size=solver.n_vars) * 0.05

        def original_objective(z):
            states, controls = solver.unpack(z)
            h = solver.dt
            integrand = np.sum(controls**2, axis=1) + self.problem.effort_weight_dq * np.sum(
                states[:, 2:] ** 2, axis=1
            )
            cost = 0.0
            for k in range(solver.N):
                idx_k, idx_mid, idx_kp1 = 2 * k, 2 * k + 1, 2 * k + 2
                cost += (h / 6.0) * (integrand[idx_k] + 4.0 * integrand[idx_mid] + integrand[idx_kp1])
            return float(cost)

        self.assertAlmostEqual(solver.objective(z), original_objective(z), places=10)

    def test_defects_match_original_loop(self):
        """The vectorized defects() equal the original's loop-based defects
        (copied inline here from hermite_simpson.py's equality_constraints)."""
        solver = ImprovedHermiteSimpsonSolver(self.problem, vectorized=False)
        z = self.rng.normal(size=solver.n_vars) * 0.05

        def original_defects(z):
            states, controls = solver.unpack(z)
            h = solver.dt
            f_vals = np.array(
                [solver.arm.state_derivative(states[i], controls[i]) for i in range(solver.total_points)]
            )
            eqs = []
            for k in range(solver.N):
                idx_k, idx_mid, idx_kp1 = 2 * k, 2 * k + 1, 2 * k + 2
                xk, xmid, xkp1 = states[idx_k], states[idx_mid], states[idx_kp1]
                fk, fmid, fkp1 = f_vals[idx_k], f_vals[idx_mid], f_vals[idx_kp1]
                eqs.append(xmid - 0.5 * (xk + xkp1) - (h / 8.0) * (fk - fkp1))
                eqs.append(xkp1 - xk - (h / 6.0) * (fk + 4.0 * fmid + fkp1))
            return np.concatenate(eqs)

        np.testing.assert_allclose(solver.defects(z), original_defects(z), atol=1e-10)

    def test_objective_gradient_matches_finite_difference(self):
        """S3(a): exact objective gradient equals a numerical estimate."""
        solver = ImprovedHermiteSimpsonSolver(self.problem)
        z = self.rng.normal(size=solver.n_vars)
        numeric = approx_derivative(solver.objective, z)
        np.testing.assert_allclose(solver.objective_grad(z), numeric, atol=1e-5)

    def test_defect_block_uses_fixed_color_group_count(self):
        """S3(b): defect rows touch 3 points each (k, mid, k+1), not 2 like
        Trapezoidal, so this needs 18 groups instead of 12, but the count
        should still be INDEPENDENT of N (checked at three different N)."""
        for N in (10, 20, 40):
            problem = small_problem()
            problem = TrajectoryProblem(
                x0=problem.x0, xf=problem.xf, duration=problem.duration,
                n_nodes=N, arm_params=problem.arm_params, tau_max=problem.tau_max,
            )
            solver = ImprovedHermiteSimpsonSolver(problem)
            defect_block = [b for b in solver.constraint_blocks() if b["fun"] == solver.defects][0]
            S = build_pattern(defect_block["row_nodes"], solver.total_points, solver.node_vars)
            self.assertEqual(len(color_columns(S)), 18, f"N={N}")

    def test_sparse_jacobian_matches_dense(self):
        """S3(b): colored Jacobian equals a normal (dense) finite-difference
        Jacobian, for every constraint block, including obstacle blocks."""
        obstacle = SCENARIOS["obstacle"]().create_problem(n_nodes=8)
        for problem in (self.problem, obstacle):
            solver = ImprovedHermiteSimpsonSolver(problem, fix_endpoints_in_bounds=False)
            z = self.rng.normal(size=solver.n_vars) * 0.05
            for blk in solver.constraint_blocks():
                S = build_pattern(blk["row_nodes"], solver.total_points, solver.node_vars)
                J_sparse = sparse_fd_jacobian(blk["fun"], z, S, color_columns(S))
                J_dense = approx_derivative(blk["fun"], z)
                np.testing.assert_allclose(J_sparse, J_dense, atol=1e-4)

    def test_bounds_lock_mesh_endpoints_only(self):
        """S1: min = max = required value for the first and last MESH point
        (index 0 and index 2N), midpoints are left free."""
        solver = ImprovedHermiteSimpsonSolver(self.problem)
        lo, hi = solver._bounds()
        lo, hi = lo.reshape(-1, 6), hi.reshape(-1, 6)
        np.testing.assert_array_equal(lo[0, :4], self.problem.x0)
        np.testing.assert_array_equal(hi[0, :4], self.problem.x0)
        np.testing.assert_array_equal(lo[-1, :4], self.problem.xf)
        np.testing.assert_array_equal(hi[-1, :4], self.problem.xf)
        # Midpoint (index 1) must NOT be locked.
        self.assertLess(lo[1, 0], hi[1, 0])

    def test_default_coarse_levels(self):
        """S5: coarse grids are N/2, N/4, ... down to >= 8 (reused from
        improved_trapezoidal.py, so this just re-confirms the import works)."""
        self.assertEqual(default_coarse_levels(30), [15])
        self.assertEqual(default_coarse_levels(40), [10, 20])
        self.assertEqual(default_coarse_levels(10), [])


class TestSameAnswerAsOriginal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = small_problem()
        cls.original = HermiteSimpsonCollocationSolver(cls.problem).solve(max_iter=200, ftol=1e-5)

    def check_same_answer(self, res):
        self.assertTrue(res.success, res.message)
        self.assertLess(res.max_constraint_violation, 1e-3)
        self.assertLess(abs(res.cost - self.original.cost) / self.original.cost, 1e-3)
        np.testing.assert_allclose(res.state[0], self.problem.x0, atol=1e-3)
        np.testing.assert_allclose(res.state[-1], self.problem.xf, atol=1e-3)

    def test_all_flags_off_reproduces_original(self):
        res = ImprovedHermiteSimpsonSolver(self.problem, **ALL_OFF).solve(max_iter=200, ftol=1e-5)
        self.check_same_answer(res)

    def test_each_improvement_alone(self):
        for flag in ALL_OFF:
            with self.subTest(flag=flag):
                res = ImprovedHermiteSimpsonSolver(self.problem, **{**ALL_OFF, flag: True}).solve(
                    max_iter=200, ftol=1e-5)
                self.check_same_answer(res)

    def test_all_improvements(self):
        res = ImprovedHermiteSimpsonSolver(self.problem).solve(max_iter=200, ftol=1e-5)
        self.check_same_answer(res)

    def test_warm_start(self):
        """S5: same answer, and the reported time/iterations include the
        coarse-grid solves."""
        res = solve_with_warm_start(self.problem, coarse_levels=[8], max_iter=200, ftol=1e-5)
        self.check_same_answer(res)
        levels = res.info["levels"]
        self.assertEqual([lv["N"] for lv in levels], [8, 15])
        self.assertAlmostEqual(res.solve_time, sum(lv["time"] for lv in levels))
        self.assertEqual(res.iterations, sum(lv["iterations"] for lv in levels))


if __name__ == "__main__":
    unittest.main()
