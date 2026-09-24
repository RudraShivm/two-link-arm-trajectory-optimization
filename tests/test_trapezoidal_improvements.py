import dataclasses
import unittest

import numpy as np

from arm_opt.analysis.reality_check import perform_reality_check
from arm_opt.dynamics.parameters import ArmParameters
from arm_opt.solvers.base import TrajectoryProblem
from arm_opt.solvers.improved_trapezoidal import (
    ImprovedTrapezoidalSolver,
    solve_with_warm_start,
    state_derivative_batch,
)
from arm_opt.dynamics.kinematics import forward_kinematics
from arm_opt.scenarios.obstacle import ObstacleScenario, segment_clearance_sq
from arm_opt.solvers.trapezoidal_utils import (
    fit_order,
    link_clearances,
    make_state_interp_fn,
    trapezoidal_interpolate,
)


def small_problem():
    params = ArmParameters(l1=1.0, l2=1.0, m1=1.0, m2=1.0, tau_max=50.0)
    return TrajectoryProblem(
        x0=np.array([0.0, 0.0, 0.0, 0.0]),
        xf=np.array([np.pi / 4, -np.pi / 4, 0.0, 0.0]),
        duration=0.8,
        n_nodes=15,
        arm_params=params,
        tau_max=50.0,
    )


class TestA1Interpolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem = small_problem()
        cls.result = ImprovedTrapezoidalSolver(cls.problem).solve(ftol=1e-10)
        cls.max_defect = cls.result.max_constraint_violation

    def interp(self, t):
        r = self.result
        return trapezoidal_interpolate(self.problem.arm, r.time, r.state, r.control, t)

    def test_solve_succeeded(self):
        self.assertTrue(self.result.success)

    def test_passes_through_every_node(self):
        x, u = self.interp(self.result.time)
        
        self.assertLessEqual(np.max(np.abs(x - self.result.state)), self.max_defect + 1e-12)
        np.testing.assert_allclose(u, self.result.control, atol=1e-12)

    def test_end_of_interval_lands_on_next_node(self):
        # Evaluate every parabola at tau = h explicitly (right end of interval k).
        r = self.result
        f = state_derivative_batch(self.problem.arm.p, r.state, r.control)
        h = np.diff(r.time)[:, None]
        x_end = r.state[:-1] + f[:-1] * h + (h / 2.0) * (f[1:] - f[:-1])
        self.assertLessEqual(np.max(np.abs(x_end - r.state[1:])), self.max_defect + 1e-12)

    def test_is_continuous_across_nodes(self):
        eps = 1e-9
        t_inner = self.result.time[1:-1]
        x_left, _ = self.interp(t_inner - eps)
        x_right, _ = self.interp(t_inner + eps)
        self.assertLess(np.max(np.abs(x_left - x_right)), self.max_defect + 1e-6)

    def test_derivative_matches_dynamics_at_nodes(self):
        # Parabola slope at t_k is f_k, by construction.
        r = self.result
        eps = 1e-6
        x_plus, _ = self.interp(r.time[:-1] + eps)
        slope = (x_plus - r.state[:-1]) / eps
        f = state_derivative_batch(self.problem.arm.p, r.state, r.control)[:-1]
        np.testing.assert_allclose(slope, f, rtol=1e-3, atol=1e-3)

    def test_reality_check_default_unchanged(self):
        rc_default = perform_reality_check(self.problem, self.result)
        rc_none = perform_reality_check(self.problem, self.result, state_interp_fn=None)
        self.assertEqual(rc_default.max_state_drift, rc_none.max_state_drift)
        self.assertEqual(rc_default.terminal_cartesian_drift, rc_none.terminal_cartesian_drift)

    def test_quadratic_hook_changes_only_max_drift(self):
        rc_lin = perform_reality_check(self.problem, self.result)
        rc_quad = perform_reality_check(
            self.problem, self.result, state_interp_fn=make_state_interp_fn(self.problem, self.result)
        )
        self.assertEqual(rc_lin.terminal_cartesian_drift, rc_quad.terminal_cartesian_drift)
        self.assertNotEqual(rc_lin.max_state_drift, rc_quad.max_state_drift)


class TestA2Convergence(unittest.TestCase):
    def test_fit_order_recovers_known_slope(self):
        hs = np.array([0.1, 0.05, 0.025, 0.0125])
        self.assertAlmostEqual(fit_order(hs, 3.0 * hs**2), 2.0, places=10)
        self.assertAlmostEqual(fit_order(hs, 0.5 * hs**4), 4.0, places=10)

    def test_trapezoidal_is_second_order(self):
        hs, state_err, final_drift = [], [], []
        for n in (20, 40, 80):
            problem = dataclasses.replace(small_problem(), n_nodes=n)
            result = solve_with_warm_start(problem, ftol=1e-10)
            self.assertTrue(result.success)
            rc = perform_reality_check(problem, result, state_interp_fn=make_state_interp_fn(problem, result))
            # Optimizer error must be far below the discretization error being measured.
            self.assertLess(result.max_constraint_violation, 1e-3 * rc.terminal_cartesian_drift)
            hs.append(problem.dt)
            state_err.append(rc.max_state_drift)
            final_drift.append(rc.terminal_cartesian_drift)
        self.assertGreater(fit_order(hs, state_err), 1.7)
        self.assertLess(fit_order(hs, state_err), 2.5)
        self.assertGreater(fit_order(hs, final_drift), 1.7)
        self.assertLess(fit_order(hs, final_drift), 2.5)


class TestA3FullBodyObstacle(unittest.TestCase):
    def test_segment_clearance_geometry(self):
        a, b, c = np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.5, 0.4])
        # Closest point is the projection (0.5, 0): distance 0.4.
        self.assertAlmostEqual(segment_clearance_sq(a, b, c, 0.1), 0.4**2 - 0.1**2)
        # Beyond the end the closest point is b itself (clamped).
        self.assertAlmostEqual(segment_clearance_sq(a, b, np.array([2.0, 0.0]), 0.0), 1.0)
        # Center on the segment: fully inside.
        self.assertAlmostEqual(segment_clearance_sq(a, b, np.array([0.3, 0.0]), 0.2), -0.04)

    def test_link_clearances_match_kinematics(self):
        rng = np.random.default_rng(1)
        params = ObstacleScenario().create_problem(n_nodes=5).arm_params
        q = rng.uniform(-np.pi, np.pi, size=(200, 2))
        center, radius = np.array([1.2, 0.3]), 0.25
        c = link_clearances(params, q, center, radius)
        for i in range(len(q)):
            elbow, hand = forward_kinematics(q[i], params)
            self.assertAlmostEqual(c["hand"][i], np.linalg.norm(hand - center) - radius)
            d1 = np.sqrt(segment_clearance_sq(np.zeros(2), elbow, center, 0.0)) - radius
            d2 = np.sqrt(segment_clearance_sq(elbow, hand, center, 0.0)) - radius
            self.assertAlmostEqual(c["link1"][i], d1)
            self.assertAlmostEqual(c["link2"][i], d2)
        self.assertTrue(np.all(c["link2"] <= c["hand"] + 1e-12))  # the hand is part of link 2

    def test_default_scenario_unchanged(self):
        problem = ObstacleScenario().create_problem(n_nodes=10)
        self.assertEqual(len(problem.path_constraints), 1)
        x = np.array([0.3, -0.2, 0.0, 0.0])
        _, p_ee = forward_kinematics(x[:2], problem.arm_params)
        expected = np.sum((p_ee - np.array([1.2, 0.0])) ** 2) - (0.35 + 0.005) ** 2
        self.assertAlmostEqual(problem.path_constraints[0](x, np.zeros(2)), expected)

    def test_full_body_solution_is_collision_free(self):
        scenario = ObstacleScenario(
            obstacle_center=ObstacleScenario.FULL_BODY_CENTER,
            obstacle_radius=ObstacleScenario.FULL_BODY_RADIUS,
            check_full_body=True,
        )
        problem = scenario.create_problem(n_nodes=20)
        result = ImprovedTrapezoidalSolver(problem).solve(max_iter=1000, ftol=1e-8)
        self.assertTrue(result.success)
        c = link_clearances(problem.arm_params, result.state, scenario.obs_center, scenario.obs_radius)
        self.assertGreaterEqual(min(c["link1"].min(), c["link2"].min()), -1e-6)


if __name__ == "__main__":
    unittest.main()
