"""Method 2: Trapezoidal Direct Collocation."""

import time
from typing import Tuple
import numpy as np
from scipy.optimize import minimize
from arm_opt.solvers.base import TrajectoryProblem, TrajectoryResult


class TrapezoidalCollocationSolver:
    """Solves the optimal control problem using Trapezoidal Direct Collocation.

    Decision variables:
        z = [x_0, u_0, x_1, u_1, ..., x_N, u_N] in R^(6 * (N + 1))
        where x_k in R^4 (states) and u_k in R^2 (controls).

    Physics enforcement:
        Defect constraints for each interval k in {0, ..., N-1}:
            Delta_k = x_{k+1} - x_k - (h/2) * (f(x_k, u_k) + f(x_{k+1}, u_{k+1})) = 0
    """

    def __init__(self, problem: TrajectoryProblem):
        self.problem = problem
        self.arm = problem.arm
        self.N = problem.n_nodes
        self.dt = problem.dt
        self.n_states = 4
        self.n_controls = 2
        self.node_vars = self.n_states + self.n_controls
        self.n_vars = (self.N + 1) * self.node_vars

    def unpack(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Unpacks 1D decision vector z into states (N+1, 4) and controls (N+1, 2)."""
        Z = z.reshape((self.N + 1, self.node_vars))
        states = Z[:, : self.n_states]
        controls = Z[:, self.n_states :]
        return states, controls

    def pack(self, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """Packs states (N+1, 4) and controls (N+1, 2) into 1D decision vector z."""
        Z = np.hstack([states, controls])
        return Z.flatten()

    def solve(self, max_iter: int = 400, ftol: float = 1e-6) -> TrajectoryResult:
        """Executes the trapezoidal collocation optimization."""
        start_time = time.perf_counter()

        # Initial guess: linear interpolation in state, gravity compensation in control
        init_states = self.problem.get_linear_state_guess()
        init_controls = np.zeros((self.N + 1, self.n_controls))
        for k in range(self.N + 1):
            init_controls[k] = np.clip(
                self.arm.gravity_vector(init_states[k, :2]),
                -self.problem.tau_max,
                self.problem.tau_max,
            )
        z0 = self.pack(init_states, init_controls)

        # Variable box bounds
        bounds = []
        p = self.problem
        for _ in range(self.N + 1):
            # State bounds: q1, q2, dq1, dq2
            bounds.append((p.q_min, p.q_max))
            bounds.append((p.q_min, p.q_max))
            bounds.append((-p.dq_max, p.dq_max))
            bounds.append((-p.dq_max, p.dq_max))
            # Control bounds: tau1, tau2
            bounds.append((-p.tau_max, p.tau_max))
            bounds.append((-p.tau_max, p.tau_max))

        # Objective function: Trapezoidal quadrature of effort
        def objective(z: np.ndarray) -> float:
            states, controls = self.unpack(z)
            h = self.dt
            # sum over intervals k: (h/2) * (L_k + L_{k+1})
            effort = np.sum(controls**2, axis=1)
            vel_reg = p.effort_weight_dq * np.sum(states[:, 2:] ** 2, axis=1)
            integrand = effort + vel_reg
            # Trapezoidal integration
            cost = 0.5 * h * (integrand[0] + 2.0 * np.sum(integrand[1:-1]) + integrand[-1])
            return float(cost)

        # Equality defect constraints & boundary conditions
        def equality_constraints(z: np.ndarray) -> np.ndarray:
            states, controls = self.unpack(z)
            h = self.dt
            eqs = []

            # 1. Boundary conditions: x_0 = x0, x_N = xf
            eqs.append(states[0] - p.x0)
            eqs.append(states[-1] - p.xf)

            # 2. Defect constraints on each interval k
            f_vals = np.zeros((self.N + 1, self.n_states))
            for k in range(self.N + 1):
                f_vals[k] = self.arm.state_derivative(states[k], controls[k])

            for k in range(self.N):
                defect_k = states[k + 1] - states[k] - 0.5 * h * (f_vals[k] + f_vals[k + 1])
                eqs.append(defect_k)

            return np.concatenate(eqs)

        constraints = [{"type": "eq", "fun": equality_constraints}]

        # Optional path constraints
        if self.problem.path_constraints:
            for path_fn in self.problem.path_constraints:

                def make_path_con(fn=path_fn):
                    def con(z: np.ndarray) -> np.ndarray:
                        states, controls = self.unpack(z)
                        vals = [fn(states[k], controls[k]) for k in range(self.N + 1)]
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
            method_name="Trapezoidal Collocation",
            success=bool(res.success),
            message=str(res.message),
            time=self.problem.time_grid,
            state=final_states,
            control=final_controls,
            cost=float(res.fun),
            solve_time=solve_time,
            iterations=int(res.nit) if hasattr(res, "nit") else 0,
            max_constraint_violation=float(max_violation),
            info={"status": res.status},
        )

