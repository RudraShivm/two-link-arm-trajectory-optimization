"""Method 1: Single Direct Shooting with 4th-Order Runge-Kutta (RK4) integration."""

import time
from typing import Tuple
import numpy as np
from scipy.optimize import minimize
from arm_opt.solvers.base import TrajectoryProblem, TrajectoryResult


class ShootingSolver:
    """Solves the optimal control problem using Single Direct Shooting.

    Decision variables:
        z = [tau_0, tau_1, ..., tau_{N-1}] in R^(2 * N)
    Physics enforcement:
        Explicit forward RK4 simulation over [0, T].
    Constraints:
        Terminal state equality: x(T) - xf = 0
        Path constraints: g(x_k, u_k) >= 0 for all k (optional)
        Box bounds on torques: -tau_max <= tau_k <= tau_max
    """

    def __init__(self, problem: TrajectoryProblem):
        self.problem = problem
        self.arm = problem.arm
        self.N = problem.n_nodes
        self.dt = problem.dt

    def _simulate_trajectory(
        self, controls: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward simulates RK4 trajectory from x0 given controls of shape (N, 2).

        Returns:
            states: Array of shape (N + 1, 4).
            controls: Array of shape (N + 1, 2) (padded terminal control for alignment).
        """
        states = np.zeros((self.N + 1, 4))
        states[0] = self.problem.x0

        for k in range(self.N):
            states[k + 1] = self.arm.rk4_step(states[k], controls[k], self.dt)

        # Pad last control for consistent array dimensions across methods
        controls_full = np.vstack([controls, controls[-1]])
        return states, controls_full

    def solve(self, max_iter: int = 300, ftol: float = 1e-6) -> TrajectoryResult:
        """Executes the shooting optimization."""
        start_time = time.perf_counter()

        # Decision variables: u_0, u_1, ..., u_{N-1} flattened -> 2 * N
        n_vars = 2 * self.N

        # Initial guess: gravity compensation along linear state interpolation
        linear_states = self.problem.get_linear_state_guess()
        u_init = np.zeros((self.N, 2))
        for k in range(self.N):
            q_k = linear_states[k, :2]
            u_init[k] = np.clip(
                self.arm.gravity_vector(q_k),
                -self.problem.tau_max,
                self.problem.tau_max,
            )
        z0 = u_init.flatten()

        # Bounds on control torques
        bounds = [(-self.problem.tau_max, self.problem.tau_max)] * n_vars

        # Objective function
        def objective(z: np.ndarray) -> float:
            u_traj = z.reshape((self.N, 2))
            states, _ = self._simulate_trajectory(u_traj)
            # Control effort + small state velocity regularization
            effort = np.sum(u_traj**2) * self.dt
            vel_reg = (
                self.problem.effort_weight_dq
                * np.sum(states[:, 2:] ** 2)
                * self.dt
            )
            return float(effort + vel_reg)

        # Terminal equality constraint: x(T) - xf = 0
        def terminal_constraint(z: np.ndarray) -> np.ndarray:
            u_traj = z.reshape((self.N, 2))
            states, _ = self._simulate_trajectory(u_traj)
            return states[-1] - self.problem.xf

        constraints = [{"type": "eq", "fun": terminal_constraint}]

        # Optional path constraints
        if self.problem.path_constraints:
            for c_idx, path_fn in enumerate(self.problem.path_constraints):

                def make_path_con(fn=path_fn):
                    def con(z: np.ndarray) -> np.ndarray:
                        u_traj = z.reshape((self.N, 2))
                        states, _ = self._simulate_trajectory(u_traj)
                        vals = [
                            fn(states[k], u_traj[min(k, self.N - 1)])
                            for k in range(self.N + 1)
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
        final_controls = res.x.reshape((self.N, 2))
        final_states, padded_controls = self._simulate_trajectory(final_controls)

        term_err = np.max(np.abs(terminal_constraint(res.x)))

        return TrajectoryResult(
            method_name="Single Shooting (RK4)",
            success=bool(res.success),
            message=str(res.message),
            time=self.problem.time_grid,
            state=final_states,
            control=padded_controls,
            cost=float(res.fun),
            solve_time=solve_time,
            iterations=int(res.nit) if hasattr(res, "nit") else 0,
            max_constraint_violation=float(term_err),
            info={"status": res.status},
        )

