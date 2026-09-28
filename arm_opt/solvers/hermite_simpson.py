import time
from typing import Tuple
import numpy as np
from scipy.optimize import minimize
from arm_opt.solvers.base import TrajectoryProblem, TrajectoryResult


class HermiteSimpsonCollocationSolver:
    """Hermite–Simpson (separated): nodes + midpoints; Hermite mid + Simpson defects."""

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
        Z = z.reshape((self.total_points, self.node_vars))
        return Z[:, : self.n_states], Z[:, self.n_states :]

    def pack(self, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        return np.hstack([states, controls]).flatten()

    def solve(self, max_iter: int = 500, ftol: float = 1e-6) -> TrajectoryResult:
        start_time = time.perf_counter()
        p = self.problem
        fine_time_grid = np.linspace(0.0, p.duration, self.total_points)

        init_states = np.linspace(p.x0, p.xf, self.total_points)
        init_controls = np.zeros((self.total_points, self.n_controls))
        for i in range(self.total_points):
            init_controls[i] = np.clip(
                self.arm.gravity_vector(init_states[i, :2]),
                -p.tau_max,
                p.tau_max,
            )
        z0 = self.pack(init_states, init_controls)

        bounds = []
        for _ in range(self.total_points):
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
            cost = 0.0
            for k in range(self.N):
                i0, im, i1 = 2 * k, 2 * k + 1, 2 * k + 2
                cost += (h / 6.0) * (
                    integrand[i0] + 4.0 * integrand[im] + integrand[i1]
                )
            return float(cost)

        def equality_constraints(z: np.ndarray) -> np.ndarray:
            states, controls = self.unpack(z)
            h = self.dt
            eqs = [states[0] - p.x0, states[-1] - p.xf]

            f_vals = np.zeros((self.total_points, self.n_states))
            for i in range(self.total_points):
                f_vals[i] = self.arm.state_derivative(states[i], controls[i])

            for k in range(self.N):
                i0, im, i1 = 2 * k, 2 * k + 1, 2 * k + 2
                xk, xmid, xkp1 = states[i0], states[im], states[i1]
                fk, fmid, fkp1 = f_vals[i0], f_vals[im], f_vals[i1]

                # x_mid = 0.5(x_k+x_{k+1}) + (h/8)(f_k - f_{k+1})
                eqs.append(xmid - 0.5 * (xk + xkp1) - (h / 8.0) * (fk - fkp1))
                # Simpson: x_{k+1}-x_k = (h/6)(f_k + 4 f_mid + f_{k+1})
                eqs.append(xkp1 - xk - (h / 6.0) * (fk + 4.0 * fmid + fkp1))

            return np.concatenate(eqs)

        constraints = [{"type": "eq", "fun": equality_constraints}]

        if self.problem.path_constraints:
            for path_fn in self.problem.path_constraints:

                def make_path_con(fn=path_fn):
                    def con(z: np.ndarray) -> np.ndarray:
                        states, controls = self.unpack(z)
                        return np.array(
                            [
                                fn(states[i], controls[i])
                                for i in range(self.total_points)
                            ]
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
