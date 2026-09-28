import time
from typing import Tuple
import numpy as np
from scipy.optimize import minimize
from arm_opt.solvers.base import TrajectoryProblem, TrajectoryResult


class TrapezoidalCollocationSolver:
    """Trapezoidal collocation: z = [x_k, u_k], defect x_{k+1}-x_k-(h/2)(f_k+f_{k+1})=0."""

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
        Z = z.reshape((self.N + 1, self.node_vars))
        return Z[:, : self.n_states], Z[:, self.n_states :]

    def pack(self, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        return np.hstack([states, controls]).flatten()

    def solve(self, max_iter: int = 400, ftol: float = 1e-6) -> TrajectoryResult:
        start_time = time.perf_counter()
        p = self.problem

        init_states = p.get_linear_state_guess()
        init_controls = np.zeros((self.N + 1, self.n_controls))
        for k in range(self.N + 1):
            init_controls[k] = np.clip(
                self.arm.gravity_vector(init_states[k, :2]),
                -p.tau_max,
                p.tau_max,
            )
        z0 = self.pack(init_states, init_controls)

        bounds = []
        for _ in range(self.N + 1):
            bounds.append((p.q_min, p.q_max))
            bounds.append((p.q_min, p.q_max))
            bounds.append((-p.dq_max, p.dq_max))
            bounds.append((-p.dq_max, p.dq_max))
            bounds.append((-p.tau_max, p.tau_max))
            bounds.append((-p.tau_max, p.tau_max))

        def objective(z: np.ndarray) -> float:
            states, controls = self.unpack(z)
            h = self.dt
            integrand = np.sum(controls**2, axis=1) + p.effort_weight_dq * np.sum(
                states[:, 2:] ** 2, axis=1
            )
            cost = 0.5 * h * (integrand[0] + 2.0 * np.sum(integrand[1:-1]) + integrand[-1])
            return float(cost)

        def equality_constraints(z: np.ndarray) -> np.ndarray:
            states, controls = self.unpack(z)
            h = self.dt
            eqs = [states[0] - p.x0, states[-1] - p.xf]

            f_vals = np.zeros((self.N + 1, self.n_states))
            for k in range(self.N + 1):
                f_vals[k] = self.arm.state_derivative(states[k], controls[k])

            for k in range(self.N):
                defect_k = states[k + 1] - states[k] - 0.5 * h * (f_vals[k] + f_vals[k + 1])
                eqs.append(defect_k)

            return np.concatenate(eqs)

        constraints = [{"type": "eq", "fun": equality_constraints}]

        if self.problem.path_constraints:
            for path_fn in self.problem.path_constraints:

                def make_path_con(fn=path_fn):
                    def con(z: np.ndarray) -> np.ndarray:
                        states, controls = self.unpack(z)
                        return np.array(
                            [fn(states[k], controls[k]) for k in range(self.N + 1)]
                        )

                    return con

                constraints.append({"type": "ineq", "fun": make_path_con()})

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
