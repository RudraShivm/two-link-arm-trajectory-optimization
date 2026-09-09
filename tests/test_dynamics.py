"""Unit tests for planar two-link manipulator dynamics and kinematics."""

import unittest
import numpy as np
from arm_opt.dynamics.parameters import ArmParameters
from arm_opt.dynamics.manipulator import TwoLinkArm
from arm_opt.dynamics.kinematics import forward_kinematics, end_effector_jacobian


class TestTwoLinkDynamics(unittest.TestCase):
    def setUp(self):
        self.params = ArmParameters(l1=1.0, l2=1.0, m1=1.0, m2=1.0)
        self.arm = TwoLinkArm(self.params)

    def test_mass_matrix_properties(self):
        """Mass matrix M(q) must be symmetric and strictly positive definite for arbitrary q."""
        test_angles = [
            np.array([0.0, 0.0]),
            np.array([np.pi / 4, -np.pi / 3]),
            np.array([-np.pi / 2, np.pi / 2]),
            np.array([1.23, -2.34]),
        ]
        for q in test_angles:
            M = self.arm.mass_matrix(q)
            # Symmetry
            np.testing.assert_allclose(M, M.T, atol=1e-12)
            # Positive definiteness (all eigenvalues strictly positive)
            eigenvalues = np.linalg.eigvals(M)
            self.assertTrue(np.all(eigenvalues > 0.0), f"Eigenvalues not positive for q={q}: {eigenvalues}")

    def test_forward_kinematics_known_poses(self):
        """Verify forward kinematics matches analytic trigonometry for canonical poses."""
        # Horizontal straight out
        _, ee0 = forward_kinematics(np.array([0.0, 0.0]), self.params)
        np.testing.assert_allclose(ee0, [2.0, 0.0], atol=1e-12)

        # Vertical straight up
        _, ee_up = forward_kinematics(np.array([np.pi / 2, 0.0]), self.params)
        np.testing.assert_allclose(ee_up, [0.0, 2.0], atol=1e-12)

        # Bent 90 degrees at elbow
        _, ee_bent = forward_kinematics(np.array([np.pi / 2, np.pi / 2]), self.params)
        np.testing.assert_allclose(ee_bent, [-1.0, 1.0], atol=1e-12)

    def test_energy_conservation_free_motion(self):
        """Under zero control torque tau=[0, 0], total mechanical energy E = T + V must be conserved."""
        x = np.array([0.2, -0.4, 0.0, 0.0])  # release from rest
        dt = 0.001
        steps = 1000  # 1.0 second simulation
        tau_zero = np.zeros(2)

        e_initial = self.arm.total_energy(x[:2], x[2:])
        for _ in range(steps):
            x = self.arm.rk4_step(x, tau_zero, dt)

        e_final = self.arm.total_energy(x[:2], x[2:])
        relative_error = abs(e_final - e_initial) / abs(e_initial)
        # RK4 with dt=1ms over 1s maintains energy within 0.01%
        self.assertLess(relative_error, 1e-4, f"Energy drift too high: relative error = {relative_error}")

    def test_jacobian_numerical_consistency(self):
        """Jacobian J(q) must match finite difference approximation."""
        q = np.array([0.5, -0.7])
        J_analytic = end_effector_jacobian(q, self.params)

        eps = 1e-6
        J_num = np.zeros((2, 2))
        _, ee0 = forward_kinematics(q, self.params)

        for i in range(2):
            q_perturbed = q.copy()
            q_perturbed[i] += eps
            _, ee_pert = forward_kinematics(q_perturbed, self.params)
            J_num[:, i] = (ee_pert - ee0) / eps

        np.testing.assert_allclose(J_analytic, J_num, atol=1e-5)


if __name__ == "__main__":
    unittest.main()

