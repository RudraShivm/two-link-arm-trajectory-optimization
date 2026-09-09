"""Nonlinear equations of motion and numerical integrators for the planar two-link manipulator."""

import numpy as np
from arm_opt.dynamics.parameters import ArmParameters, DEFAULT_PARAMS


class TwoLinkArm:
    """Rigid body Lagrangian dynamics model for a planar two-link manipulator."""

    def __init__(self, params: ArmParameters = DEFAULT_PARAMS):
        self.p = params

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Computes the 2x2 symmetric, positive-definite inertia matrix M(q)."""
        theta2 = q[1]
        c2 = np.cos(theta2)

        p = self.p
        m2_l1_r2 = p.m2 * p.l1 * p.r2

        m11 = (
            p.m1 * (p.r1**2)
            + p.I1
            + p.m2 * (p.l1**2 + p.r2**2 + 2.0 * p.l1 * p.r2 * c2)
            + p.I2
        )
        m12 = p.m2 * (p.r2**2 + p.l1 * p.r2 * c2) + p.I2
        m22 = p.m2 * (p.r2**2) + p.I2

        return np.array([[m11, m12], [m12, m22]], dtype=float)

    def coriolis_vector(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """Computes the 2-element Coriolis and centrifugal force vector C(q, dq) * dq."""
        theta2 = q[1]
        dtheta1 = dq[0]
        dtheta2 = dq[1]

        h = self.p.m2 * self.p.l1 * self.p.r2 * np.sin(theta2)

        c1 = -h * (2.0 * dtheta1 * dtheta2 + dtheta2**2)
        c2 = h * (dtheta1**2)

        return np.array([c1, c2], dtype=float)

    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """Computes the 2-element gravitational torque vector g(q) = dV/dq."""
        theta1 = q[0]
        theta12 = q[0] + q[1]

        c1 = np.cos(theta1)
        c12 = np.cos(theta12)

        p = self.p
        g1 = (p.m1 * p.r1 + p.m2 * p.l1) * p.g * c1 + p.m2 * p.r2 * p.g * c12
        g2 = p.m2 * p.r2 * p.g * c12

        return np.array([g1, g2], dtype=float)

    def potential_energy(self, q: np.ndarray) -> float:
        """Computes total gravitational potential energy V(q)."""
        theta1 = q[0]
        theta12 = q[0] + q[1]

        p = self.p
        v1 = p.m1 * p.g * p.r1 * np.sin(theta1)
        v2 = p.m2 * p.g * (p.l1 * np.sin(theta1) + p.r2 * np.sin(theta12))
        return float(v1 + v2)

    def kinetic_energy(self, q: np.ndarray, dq: np.ndarray) -> float:
        """Computes total kinetic energy T(q, dq) = 1/2 * dq^T * M(q) * dq."""
        M = self.mass_matrix(q)
        return float(0.5 * dq.dot(M.dot(dq)))

    def total_energy(self, q: np.ndarray, dq: np.ndarray) -> float:
        """Computes total mechanical energy E = T + V."""
        return self.kinetic_energy(q, dq) + self.potential_energy(q)

    def forward_dynamics(
        self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray
    ) -> np.ndarray:
        """Computes joint angular accelerations ddq = M(q)^(-1) * (tau - C(q, dq)*dq - g(q))."""
        M = self.mass_matrix(q)
        C_dq = self.coriolis_vector(q, dq)
        g = self.gravity_vector(q)
        rhs = tau - C_dq - g
        return np.linalg.solve(M, rhs)

    def state_derivative(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Continuous state derivative dx/dt = f(x, u) for state x = [q; dq].

        Args:
            x: State vector of shape (4,), [theta1, theta2, dtheta1, dtheta2].
            u: Control input vector of shape (2,), [tau1, tau2].

        Returns:
            dx/dt of shape (4,), [dtheta1, dtheta2, ddtheta1, ddtheta2].
        """
        q = x[:2]
        dq = x[2:]
        ddq = self.forward_dynamics(q, dq, u)
        return np.concatenate([dq, ddq])

    def rk4_step(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        """Performs one explicit classical 4th-order Runge-Kutta integration step."""
        k1 = self.state_derivative(x, u)
        k2 = self.state_derivative(x + 0.5 * dt * k1, u)
        k3 = self.state_derivative(x + 0.5 * dt * k2, u)
        k4 = self.state_derivative(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

