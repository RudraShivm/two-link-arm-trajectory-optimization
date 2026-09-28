import unittest
import numpy as np
from arm_opt.dynamics.parameters import ArmParameters
from arm_opt.dynamics.manipulator import TwoLinkArm
from arm_opt.dynamics.kinematics import forward_kinematics, end_effector_jacobian


class TestTwoLinkDynamics(unittest.TestCase):
    def setUp(self):
        self.params = ArmParameters(l1=1.0, l2=1.0, m1=1.0, m2=1.0)
        self.arm = TwoLinkArm(self.params)

    def test_mass_matrix_symmetric_pd(self):
        for q in [
            np.array([0.0, 0.0]),
            np.array([np.pi / 4, -np.pi / 3]),
            np.array([-np.pi / 2, np.pi / 2]),
            np.array([1.23, -2.34]),
        ]:
            M = self.arm.mass_matrix(q)
            np.testing.assert_allclose(M, M.T, atol=1e-12)
            self.assertTrue(np.all(np.linalg.eigvals(M) > 0.0))

    def test_forward_kinematics_poses(self):
        _, ee0 = forward_kinematics(np.array([0.0, 0.0]), self.params)
        np.testing.assert_allclose(ee0, [2.0, 0.0], atol=1e-12)

        _, ee_up = forward_kinematics(np.array([np.pi / 2, 0.0]), self.params)
        np.testing.assert_allclose(ee_up, [0.0, 2.0], atol=1e-12)

        _, ee_bent = forward_kinematics(np.array([np.pi / 2, np.pi / 2]), self.params)
        np.testing.assert_allclose(ee_bent, [-1.0, 1.0], atol=1e-12)

    def test_gravity_matches_potential_gradient(self):
        q = np.array([0.4, -0.7])
        g_analytic = self.arm.gravity_vector(q)
        eps = 1e-7
        g_num = np.zeros(2)
        v0 = self.arm.potential_energy(q)
        for i in range(2):
            qp = q.copy()
            qp[i] += eps
            g_num[i] = (self.arm.potential_energy(qp) - v0) / eps
        np.testing.assert_allclose(g_analytic, g_num, rtol=1e-5, atol=1e-5)

    def test_coriolis_skew_symmetry(self):
        # Christoffel form of C satisfies skew-symmetry of Mdot - 2C
        q = np.array([0.3, -0.5])
        dq = np.array([1.2, -0.8])
        eps = 1e-7

        M = self.arm.mass_matrix(q)
        Mdot = np.zeros((2, 2))
        for i in range(2):
            qp = q.copy()
            qp[i] += eps
            Mdot += ((self.arm.mass_matrix(qp) - M) / eps) * dq[i]

        h = self.params.m2 * self.params.l1 * self.params.r2 * np.sin(q[1])
        C = np.array(
            [
                [-h * dq[1], -h * (dq[0] + dq[1])],
                [h * dq[0], 0.0],
            ],
            dtype=float,
        )
        np.testing.assert_allclose(
            C.dot(dq), self.arm.coriolis_vector(q, dq), atol=1e-12
        )
        S = Mdot - 2.0 * C
        np.testing.assert_allclose(S + S.T, 0.0, atol=1e-5)

    def test_energy_conservation_free_motion(self):
        x = np.array([0.2, -0.4, 0.0, 0.0])
        dt = 0.001
        tau_zero = np.zeros(2)
        e0 = self.arm.total_energy(x[:2], x[2:])
        for _ in range(1000):
            x = self.arm.rk4_step(x, tau_zero, dt)
        e1 = self.arm.total_energy(x[:2], x[2:])
        self.assertLess(abs(e1 - e0) / abs(e0), 1e-4)

    def test_jacobian_finite_difference(self):
        q = np.array([0.5, -0.7])
        J = end_effector_jacobian(q, self.params)
        eps = 1e-6
        J_num = np.zeros((2, 2))
        _, ee0 = forward_kinematics(q, self.params)
        for i in range(2):
            qp = q.copy()
            qp[i] += eps
            _, ee = forward_kinematics(qp, self.params)
            J_num[:, i] = (ee - ee0) / eps
        np.testing.assert_allclose(J, J_num, atol=1e-5)

    def test_equations_of_motion_residual(self):
        q = np.array([0.6, -0.3])
        dq = np.array([0.5, -1.1])
        tau = np.array([3.0, -1.5])
        ddq = self.arm.forward_dynamics(q, dq, tau)
        residual = (
            self.arm.mass_matrix(q).dot(ddq)
            + self.arm.coriolis_vector(q, dq)
            + self.arm.gravity_vector(q)
            - tau
        )
        np.testing.assert_allclose(residual, 0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
