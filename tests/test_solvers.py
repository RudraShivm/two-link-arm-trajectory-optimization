import unittest
import numpy as np
from arm_opt.dynamics.parameters import ArmParameters
from arm_opt.dynamics.manipulator import TwoLinkArm
from arm_opt.solvers.base import TrajectoryProblem
from arm_opt.solvers.shooting import ShootingSolver
from arm_opt.solvers.trapezoidal import TrapezoidalCollocationSolver
from arm_opt.solvers.hermite_simpson import HermiteSimpsonCollocationSolver
from arm_opt.analysis.metrics import compute_metrics
from arm_opt.analysis.reality_check import perform_reality_check
from arm_opt.scenarios.rest_to_rest import RestToRestScenario


class TestSolvers(unittest.TestCase):
    def setUp(self):
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

    def test_shooting_converges(self):
        result = ShootingSolver(self.problem).solve(max_iter=150, ftol=1e-5)
        self.assertTrue(result.success, result.message)
        self.assertLess(result.max_constraint_violation, 1e-2)
        np.testing.assert_allclose(result.state[0], self.x0, atol=1e-5)
        np.testing.assert_allclose(result.state[-1], self.xf, atol=1e-2)

    def test_trapezoidal_converges(self):
        result = TrapezoidalCollocationSolver(self.problem).solve(max_iter=150, ftol=1e-5)
        self.assertTrue(result.success, result.message)
        self.assertLess(result.max_constraint_violation, 1e-3)
        np.testing.assert_allclose(result.state[0], self.x0, atol=1e-3)
        np.testing.assert_allclose(result.state[-1], self.xf, atol=1e-3)

    def test_hermite_simpson_converges(self):
        result = HermiteSimpsonCollocationSolver(self.problem).solve(
            max_iter=150, ftol=1e-5
        )
        self.assertTrue(result.success, result.message)
        self.assertLess(result.max_constraint_violation, 1e-3)
        np.testing.assert_allclose(result.state[0], self.x0, atol=1e-3)
        np.testing.assert_allclose(result.state[-1], self.xf, atol=1e-3)

    def test_trapezoidal_defect_formula(self):
        # On an exact RK4 trajectory with constant u, trapezoidal residual should be small at fine dt
        arm = TwoLinkArm(self.params)
        N = 40
        dt = 0.02
        u = np.array([2.0, -1.0])
        x = np.zeros((N + 1, 4))
        x[0] = self.x0
        for k in range(N):
            x[k + 1] = arm.rk4_step(x[k], u, dt)

        max_defect = 0.0
        for k in range(N):
            fk = arm.state_derivative(x[k], u)
            fkp1 = arm.state_derivative(x[k + 1], u)
            defect = x[k + 1] - x[k] - 0.5 * dt * (fk + fkp1)
            max_defect = max(max_defect, float(np.max(np.abs(defect))))
        self.assertLess(max_defect, 5e-3)

    def test_hermite_simpson_defect_formula(self):
        # Constant-u RK4 path: Hermite + Simpson residuals vanish as h shrinks
        arm = TwoLinkArm(self.params)
        u = np.array([1.5, -0.5])
        residuals = []
        for h in (0.08, 0.04, 0.02):
            half_steps = 40
            sub = (h / 2.0) / half_steps
            x0 = self.x0.copy()
            x = x0.copy()
            for _ in range(half_steps):
                x = arm.rk4_step(x, u, sub)
            x_mid = x.copy()
            for _ in range(half_steps):
                x = arm.rk4_step(x, u, sub)
            x1 = x
            f0 = arm.state_derivative(x0, u)
            f1 = arm.state_derivative(x1, u)
            fmid = arm.state_derivative(x_mid, u)
            hermite = x_mid - 0.5 * (x0 + x1) - (h / 8.0) * (f0 - f1)
            simpson = x1 - x0 - (h / 6.0) * (f0 + 4.0 * fmid + f1)
            residuals.append(max(np.max(np.abs(hermite)), np.max(np.abs(simpson))))
        self.assertLess(residuals[1], residuals[0])
        self.assertLess(residuals[2], residuals[1])
        self.assertLess(residuals[-1], 1e-4)

    def test_metrics_and_reality_check(self):
        result = TrapezoidalCollocationSolver(self.problem).solve(max_iter=150, ftol=1e-5)
        self.assertTrue(result.success, result.message)
        metrics = compute_metrics(self.problem, result)
        self.assertGreaterEqual(metrics.total_effort, 0.0)
        self.assertLess(metrics.terminal_ee_error_meters, 0.05)
        rc = perform_reality_check(self.problem, result, num_eval_points=50)
        self.assertGreaterEqual(rc.max_cartesian_drift, 0.0)
        self.assertEqual(len(rc.t_sim), 50)

    def test_rest_to_rest_scenario_builds(self):
        scenario = RestToRestScenario()
        problem = scenario.create_problem(n_nodes=10)
        self.assertEqual(problem.n_nodes, 10)
        self.assertAlmostEqual(problem.duration, 1.0)
        np.testing.assert_allclose(problem.x0, [0, 0, 0, 0])
        np.testing.assert_allclose(problem.xf, [np.pi / 2, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
