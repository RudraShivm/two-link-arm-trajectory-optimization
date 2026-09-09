"""Method 3: Hermite-Simpson Direct Collocation (Separated Form)."""

import time
from typing import Tuple
import numpy as np
from scipy.optimize import minimize
from arm_opt.solvers.base import TrajectoryProblem, TrajectoryResult


class HermiteSimpsonCollocationSolver:
    """Solves the optimal control problem using Hermite-Simpson Direct Collocation.

    Separated form:
        Decision variables are defined at both grid nodes t_k and interior midpoints t_{k+1/2}.
        Total points K = 2 * N + 1.
        z in R^((2 * N + 1) * 6)
        where even indices 2*k are mesh points and odd indices 2*k + 1 are midpoints.

    Physics enforcement:
        1. Hermite interpolant defect:
            Delta_{k}^{mid} = x_{mid} - 0.5 * (x_k + x_{k+1}) - (h/8) * (f_k - f_{k+1}) = 0
        2. Simpson quadrature defect:
            Delta_{k}^{dyn} = x_{k+1} - x_k - (h/6) * (f_k + 4 * f_{mid} + f_{k+1}) = 0
    """

    def __init__(self, problem: TrajectoryProblem):
        self.problem = problem
        self.arm = problem.arm
        self.N = problem.n_nodes
        self.dt = problem.dt
        self.n_states = 4
        self.n_controls = 2
        self.node_vars = self.n_states + self.n_controls
        self.total_points = 2 * self.N + 1
        self.n_vars = self.total_points * self.node_vars

    def unpack(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unpacks 1D decision vector z into states (2N+1, 4) and controls (2N+1, 2)."""
        Z = z.reshape((self.total_points, self.node_vars))
        states = Z[:, : self.n_states]
        controls = Z[:, self.n_states :]
        return states, controls

    def pack(self, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """Packs states (2N+1, 4) and controls (2N+1, 2) into 1D decision vector z."""
        Z = np.hstack([states, controls])
        return Z.flatten()

    def solve(self, max_iter: int = 500, ftol: float = 1e-6) -> TrajectoryResult:
        """Executes the Hermite-Simpson collocation optimization."""
        start_time = time.perf_counter()
        p = self.problem

        # Time grid for all 2N + 1 points
        fine_time_grid = np.linspace(0.0, p.duration, self.total_points)

        # Initial guess: linear interpolation for all 2N + 1 points
        init_states = np.linspace(p.x0, p.xf, self.total_points)
        init_controls = np.zeros((self.total_points, self.n_controls))
        for i in range(self.total_points):
            init_controls[i] = np.clip(
                self.arm.gravity_vector(init_states[i, :2]),
                -p.tau_max,
                p.tau_max,
            )
        z0 = self.pack(init_states, init_controls)

        # Variable box bounds
        bounds = []
        for _ in range(self.total_points):
            # State bounds: q1, q2, dq1, dq2
            bounds.append((p.q_min, p.q_max))
            bounds.append((p.q_min, p.q_max))
            bounds.append((-p.dq_max, p.dq_max))
            bounds.append((-p.dq_max, p.dq_max))
            # Control bounds: tau1, tau2
            bounds.append((-p.tau_max, p.tau_max))
            bounds.append((-p.tau_max, p.tau_max))

        # Objective function: Simpson quadrature
        def objective(z: np.ndarray) -> float:
            states, controls = self.unpack(z)
            h = self.dt
            effort = np.sum(controls**2, axis=1)
            vel_reg = p.effort_weight_dq * np.sum(states[:, 2:] ** 2, axis=1)
            integrand = effort + vel_reg

            cost = 0.0
            for k in range(self.N):
                idx_k = 2 * k
                idx_mid = 2 * k + 1
                idx_kp1 = 2 * k + 2
                cost += (h / 6.0) * (
                    integrand[idx_k]
                    + 4.0 * integrand[idx_mid]
                    + integrand[idx_kp1]
                )
            return float(cost)

        # Equality defect constraints & boundary conditions
        def equality_constraints(z: np.ndarray) -> np.ndarray:
            states, controls = self.unpack(z)
            h = self.dt
            eqs = []

            # 1. Boundary conditions at ends
            eqs.append(states[0] - p.x0)
            eqs.append(states[-1] - p.xf)

            # Evaluate continuous state derivative f at all 2N+1 points
            f_vals = np.zeros((self.total_points, self.n_states))
            for i in range(self.total_points):
                f_vals[i] = self.arm.state_derivative(states[i], controls[i])

            # 2. Defect constraints for each interval k
            for k in range(self.N):
                idx_k = 2 * k
                idx_mid = 2 * k + 1
                idx_kp1 = 2 * k + 2

                xk = states[idx_k]
                xmid = states[idx_mid]
                xkp1 = states[idx_kp1]

                fk = f_vals[idx_k]
                fmid = f_vals[idx_mid]
                fkp1 = f_vals[idx_kp1]

                # Hermite interpolant defect (midpoint state constraint)
                defect_hermite = xmid - 0.5 * (xk + xkp1) - (h / 8.0) * (fk - fkp1)
                eqs.append(defect_hermite)

                # Simpson quadrature defect (system dynamics over interval)
                defect_simpson = xkp1 - xk - (h / 6.0) * (fk + 4.0 * fmid + fkp1)
                eqs.append(defect_simpson)

            return np.concatenate(eqs)

        constraints = [{"type": "eq", "fun": equality_constraints}]

        # Optional path constraints
        if self.problem.path_constraints:
            for path_fn in self.problem.path_constraints:

                def make_path_con(fn=path_fn):
                    def con(z: np.ndarray) -> np.ndarray:
                        states, controls = self.unpack(z)
                        vals = [
                            fn(states[i], controls[i])
                            for i in range(self.total_points)
                        ]
                        return np.array(vals)

                    return con

                constraints.append({"type": "ineq", "fun": make_path_con()})

        # Run optimizer
        res = minimize(
            fun=objective,
            x0=z0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": max_iter, "ftol": ftol, "disp": False},
        )

        solve_time = time.perf_counter() - start_time
        final_states, final_controls = self.unpack(res.x)

        max_violation = np.max(np.abs(equality_constraints(res.x)))

        return TrajectoryResult(
            method_name="Hermite-Simpson Collocation",
            success=bool(res.success),
            message=str(res.message),
            time=fine_time_grid,
            state=final_states,
            control=final_controls,
            cost=float(res.fun),
            solve_time=solve_time,
            iterations=int(res.nit) if hasattr(res, "nit") else 0,
            max_constraint_violation=float(max_violation),
            info={"status": res.status, "total_nodes": self.total_points},
        )

