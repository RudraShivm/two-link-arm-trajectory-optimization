import numpy as np
from arm_opt.dynamics.parameters import ArmParameters, DEFAULT_PARAMS


class TwoLinkArm:
    def __init__(self, params: ArmParameters = DEFAULT_PARAMS):
        self.p = params

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        theta2 = q[1]
        c2 = np.cos(theta2)
        p = self.p

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
        # C(q,dq)dq with h = m2*l1*r2*sin(theta2)
        h = self.p.m2 * self.p.l1 * self.p.r2 * np.sin(q[1])
        dtheta1, dtheta2 = dq[0], dq[1]
        c1 = -h * (2.0 * dtheta1 * dtheta2 + dtheta2**2)
        c2 = h * (dtheta1**2)
        return np.array([c1, c2], dtype=float)

    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        p = self.p
        c1 = np.cos(q[0])
        c12 = np.cos(q[0] + q[1])
        g1 = (p.m1 * p.r1 + p.m2 * p.l1) * p.g * c1 + p.m2 * p.r2 * p.g * c12
        g2 = p.m2 * p.r2 * p.g * c12
        return np.array([g1, g2], dtype=float)

    def potential_energy(self, q: np.ndarray) -> float:
        p = self.p
        v1 = p.m1 * p.g * p.r1 * np.sin(q[0])
        v2 = p.m2 * p.g * (p.l1 * np.sin(q[0]) + p.r2 * np.sin(q[0] + q[1]))
        return float(v1 + v2)

    def kinetic_energy(self, q: np.ndarray, dq: np.ndarray) -> float:
        M = self.mass_matrix(q)
        return float(0.5 * dq.dot(M.dot(dq)))

    def total_energy(self, q: np.ndarray, dq: np.ndarray) -> float:
        return self.kinetic_energy(q, dq) + self.potential_energy(q)

    def forward_dynamics(
        self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray
    ) -> np.ndarray:
        M = self.mass_matrix(q)
        rhs = tau - self.coriolis_vector(q, dq) - self.gravity_vector(q)
        return np.linalg.solve(M, rhs)

    def state_derivative(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        q, dq = x[:2], x[2:]
        ddq = self.forward_dynamics(q, dq, u)
        return np.concatenate([dq, ddq])

    def rk4_step(self, x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        k1 = self.state_derivative(x, u)
        k2 = self.state_derivative(x + 0.5 * dt * k1, u)
        k3 = self.state_derivative(x + 0.5 * dt * k2, u)
        k4 = self.state_derivative(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
