"""Method 3 (Improved): Hermite-Simpson Direct Collocation with speed improvements S2/S3/S5.

This file is a faster version of `hermite_simpson.py`. The MATH PROBLEM IS THE
SAME (same unknowns, same Simpson-quadrature objective, same Hermite/Simpson
defects), so the answer should be the same too. Only HOW we hand the problem
to the optimizer changes.

The original `hermite_simpson.py` is left untouched so it can serve as the
baseline for before/after comparisons, exactly like `trapezoidal.py` did for
`improved_trapezoidal.py`.

This module reuses three helpers directly from `improved_trapezoidal.py`
instead of re-writing them, because they were written generically:

    state_derivative_batch  : does not know or care whether a "node" is a
                              mesh point or a midpoint, it just needs arrays
                              of states/controls, so it applies unchanged.
    build_pattern / color_columns / sparse_fd_jacobian
                            : only need, for every constraint row, which
                              node indices it touches. Hermite-Simpson rows
                              touch THREE nodes (k, mid, k+1) instead of two,
                              but the functions handle any row width already.

Improvements ported over (S1 was already "free" here: the original file
never had endpoint equality constraints as a separate cost, it is included
for symmetry with the trapezoidal file and because it still removes 8 rows):

    S1  fix_endpoints_in_bounds : lock start/end mesh states with bounds
                                  (min = max) instead of 8 equality
                                  constraints.
    S2  vectorized              : compute f(x, u) for ALL 2N+1 points
                                  (mesh nodes AND midpoints) in one numpy
                                  call instead of a Python loop.
    S3  sparse_jacobian         : exact gradient of the Simpson-quadrature
                                  objective, plus a "colored" finite
                                  difference constraint Jacobian.
    S5  warm start              : `solve_with_warm_start()` (this file)
                                  solves a coarse grid first and uses that
                                  answer, interpolated onto both mesh AND
                                  midpoint points, as the initial guess.

Quick reminder of the problem (see hermite_simpson.py's docstring):
    unknowns  z = [x_0, u_0, x_mid0, u_mid0, x_1, u_1, ..., x_N, u_N]
              total_points = 2N + 1, node_vars = 6 each
    minimize  effort = Simpson-quadrature integral of (tau^2 + 0.001 * dq^2)
    such that x_0 = A, x_N = B  (mesh endpoints only)
              Delta_k^mid = x_mid - 0.5*(x_k+x_{k+1}) - (h/8)*(f_k - f_{k+1}) = 0
              Delta_k^dyn = x_{k+1} - x_k - (h/6)*(f_k + 4*f_mid + f_{k+1}) = 0
              bounds on every variable, optional obstacle constraints >= 0
"""

import dataclasses
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from arm_opt.solvers.base import TrajectoryProblem, TrajectoryResult
from arm_opt.solvers.improved_trapezoidal import (
    state_derivative_batch,   # S2 helper (physics for many nodes at once)
    build_pattern,            # S3 helper (sparsity pattern from row_nodes)
    color_columns,            # S3 helper (greedy column grouping)
    sparse_fd_jacobian,       # S3 helper (colored finite-difference Jacobian)
)


class ImprovedHermiteSimpsonSolver:
    """Hermite-Simpson Direct Collocation with speed improvements S1-S3.

    Same decision variables, objective and defects as
    HermiteSimpsonCollocationSolver. All improvements are ON by default.
    Turn them off one by one to measure how much each one helps, exactly
    like ImprovedTrapezoidalSolver.

    For S5 (warm start) use the function `solve_with_warm_start()` below.
    """

    def __init__(
        self,
        problem: TrajectoryProblem,
        fix_endpoints_in_bounds: bool = True,  # S1
        vectorized: bool = True,  # S2
        sparse_jacobian: bool = True,  # S3
    ):
        self.problem = problem
        self.arm = problem.arm
        self.N = problem.n_nodes  # number of intervals
        self.dt = problem.dt  # h = T / N
        self.n_states = 4
        self.n_controls = 2
        self.node_vars = self.n_states + self.n_controls  # 6
        self.total_points = 2 * self.N + 1  # mesh nodes + midpoints
        self.n_vars = self.total_points * self.node_vars

        self.fix_endpoints_in_bounds = fix_endpoints_in_bounds
        self.vectorized = vectorized
        self.sparse_jacobian = sparse_jacobian

    # ------------------------------------------------------------------ helpers

    def unpack(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Flat vector z -> states (2N+1, 4) and controls (2N+1, 2)."""
        Z = z.reshape((self.total_points, self.node_vars))
        return Z[:, : self.n_states], Z[:, self.n_states :]

    def pack(self, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """States (2N+1, 4) and controls (2N+1, 2) -> flat vector z."""
        return np.hstack([states, controls]).flatten()

    def dynamics_at_points(self, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """f(x_i, u_i) for every one of the 2N+1 points (mesh AND midpoints), shape (2N+1, 4).

        This is the same helper used by the improved trapezoidal solver: it
        does not distinguish mesh nodes from midpoints, it just needs arrays
        of states/controls, so it is reused unchanged.
        """
        if self.vectorized:
            # S2: one numpy call for all 2N+1 points at once
            return state_derivative_batch(self.problem.arm_params, states, controls)
        # Original behaviour: Python loop, one point at a time
        return np.array(
            [self.arm.state_derivative(states[i], controls[i]) for i in range(self.total_points)]
        )

    def defects(self, z: np.ndarray) -> np.ndarray:
        """All Hermite + Simpson defects, flattened to shape (8N,).

        For interval k (k = 0 .. N-1), using even indices for mesh nodes and
        odd indices for midpoints:
            idx_k, idx_mid, idx_kp1 = 2k, 2k+1, 2k+2

            Delta_k^mid = x_mid - 0.5*(x_k+x_{k+1}) - (h/8)*(f_k - f_{k+1})
            Delta_k^dyn = x_{k+1} - x_k - (h/6)*(f_k + 4*f_mid + f_{k+1})

        Instead of looping over k, we use strided array slices:
            mesh nodes    = states[0::2]   ->  x_0, x_1, ..., x_N   (N+1 rows)
            midpoints     = states[1::2]   ->  x_mid0, ..., x_mid(N-1)  (N rows)
        """
        states, controls = self.unpack(z)
        f = self.dynamics_at_points(states, controls)

        x_mesh, f_mesh = states[0::2], f[0::2]      # (N+1, 4) each
        x_mid, f_mid = states[1::2], f[1::2]        # (N, 4) each

        xk, xkp1 = x_mesh[:-1], x_mesh[1:]          # (N, 4) each
        fk, fkp1 = f_mesh[:-1], f_mesh[1:]          # (N, 4) each

        d_hermite = x_mid - 0.5 * (xk + xkp1) - (self.dt / 8.0) * (fk - fkp1)
        d_simpson = xkp1 - xk - (self.dt / 6.0) * (fk + 4.0 * f_mid + fkp1)

        # Interleave the two defect blocks per interval, same row order the
        # original solver produces (Hermite defect, then Simpson defect, for
        # k=0, then k=1, ...), so max_violation() etc. stay comparable.
        out = np.empty((self.N, 2, self.n_states))
        out[:, 0, :] = d_hermite
        out[:, 1, :] = d_simpson
        return out.flatten()

    def max_violation(self, z: np.ndarray) -> float:
        """Worst violation of start/end conditions and defects.

        Uses exactly the same definition as the original solver, so the two
        solvers' numbers can be compared directly.
        """
        states, _ = self.unpack(z)
        p = self.problem
        all_eqs = np.concatenate([states[0] - p.x0, states[-1] - p.xf, self.defects(z)])
        return float(np.max(np.abs(all_eqs)))

    # ------------------------------------------------------ building the problem

    def _initial_guess(self, initial_guess) -> np.ndarray:
        """Starting point for the optimizer (flat vector z0).

        - No guess given: same as the original solver (straight line from A
          to B for the states, "just hold the arm against gravity" for the
          torques), evaluated at all 2N+1 points including midpoints.
        - Guess given (S5, warm start): a previous solution (t_old, X_old,
          U_old), possibly on a different grid, is stretched onto our fine
          time grid (mesh points AND midpoints) by linear interpolation.
        """
        p = self.problem
        fine_time_grid = np.linspace(0.0, p.duration, self.total_points)
        if initial_guess is None:
            states = np.linspace(p.x0, p.xf, self.total_points)
            controls = np.array(
                [np.clip(self.arm.gravity_vector(states[i, :2]), -p.tau_max, p.tau_max)
                 for i in range(self.total_points)]
            )
        else:
            t_old, X_old, U_old = initial_guess
            states = np.column_stack(
                [np.interp(fine_time_grid, t_old, X_old[:, i]) for i in range(4)]
            )
            controls = np.column_stack(
                [np.interp(fine_time_grid, t_old, U_old[:, i]) for i in range(2)]
            )
            # Endpoints must match exactly (important when S1 locks them).
            states[0], states[-1] = p.x0, p.xf
        return self.pack(states, controls)

    def _bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Lower and upper bound for every variable, as two arrays of length n_vars."""
        p = self.problem
        node_lo = np.array([p.q_min, p.q_min, -p.dq_max, -p.dq_max, -p.tau_max, -p.tau_max])
        node_hi = np.array([p.q_max, p.q_max, p.dq_max, p.dq_max, p.tau_max, p.tau_max])
        lo = np.tile(node_lo, (self.total_points, 1))
        hi = np.tile(node_hi, (self.total_points, 1))

        if self.fix_endpoints_in_bounds:
            # S1: lock the 4 states of the first and last MESH point
            # (index 0 and index 2N; midpoints are never boundary conditions).
            lo[0, :4] = hi[0, :4] = p.x0
            lo[-1, :4] = hi[-1, :4] = p.xf
        return lo.flatten(), hi.flatten()

    def objective(self, z: np.ndarray) -> float:
        """Effort = Simpson-quadrature integral of (tau1^2+tau2^2 + c*(dq1^2+dq2^2)).

        Identical to the original solver, just vectorized.
        Composite Simpson: (h/6) * sum_k [L_k + 4*L_mid_k + L_{k+1}].
        """
        states, controls = self.unpack(z)
        L = np.sum(controls**2, axis=1) + self.problem.effort_weight_dq * np.sum(states[:, 2:] ** 2, axis=1)
        L_mesh, L_mid = L[0::2], L[1::2]
        cost = (self.dt / 6.0) * np.sum(L_mesh[:-1] + 4.0 * L_mid + L_mesh[1:])
        return float(cost)

    def objective_grad(self, z: np.ndarray) -> np.ndarray:
        """S3(a): EXACT gradient of the Simpson-quadrature objective.

        Every mesh node k (0 < k < N) appears in two intervals (as the
        "k+1" end of interval k-1 and the "k" end of interval k), so it
        gets weight 2. The two mesh endpoints (k=0 and k=N) appear in only
        one interval each, weight 1. Every midpoint appears in exactly one
        interval, weight 4 (Simpson's middle coefficient).

            d cost/d tau_k    = h * w_k * tau_k   (mesh, w in {1,2})
            d cost/d tau_mid  = h * (4/6*... ) -> see derivation below
        """
        states, controls = self.unpack(z)
        h = self.dt
        w_mesh = np.full(self.N + 1, 2.0)
        w_mesh[0] = w_mesh[-1] = 1.0
        w_mid = np.full(self.N, 4.0)

        G = np.zeros((self.total_points, self.node_vars))
        c = self.problem.effort_weight_dq

        # Mesh points: indices 0, 2, 4, ..., 2N
        G[0::2, 2:4] = (h / 6.0) * w_mesh[:, None] * 2.0 * c * states[0::2, 2:4]
        G[0::2, 4:6] = (h / 6.0) * w_mesh[:, None] * 2.0 * controls[0::2]
        # Midpoints: indices 1, 3, 5, ..., 2N-1
        G[1::2, 2:4] = (h / 6.0) * w_mid[:, None] * 2.0 * c * states[1::2, 2:4]
        G[1::2, 4:6] = (h / 6.0) * w_mid[:, None] * 2.0 * controls[1::2]
        return G.flatten()

    def constraint_blocks(self) -> List[Dict]:
        """All constraints, as a list of "blocks" (same shape as improved_trapezoidal.py).

        Each block is a dict:
            "type":      "eq" (must be = 0) or "ineq" (must be >= 0)
            "fun":       function z -> array of numbers
            "row_nodes": for every row of "fun", the POINT indices (0..2N)
                         that row depends on (needed by the sparse Jacobian).

        Note row_nodes here use indices into the 2N+1-point array, not the
        N+1 mesh-only array, since defects touch a midpoint too.
        """
        p = self.problem
        N = self.N
        blocks = []

        if not self.fix_endpoints_in_bounds:
            blocks.append({
                "type": "eq",
                "fun": lambda z: self.unpack(z)[0][0] - p.x0,
                "row_nodes": [(0,)] * 4,
            })
            blocks.append({
                "type": "eq",
                "fun": lambda z: self.unpack(z)[0][-1] - p.xf,
                "row_nodes": [(2 * N,)] * 4,
            })

        # Defects: 8 rows per interval k (4 Hermite + 4 Simpson), each row
        # depends on the three points (2k, 2k+1, 2k+2).
        defect_row_nodes = []
        for k in range(N):
            triple = (2 * k, 2 * k + 1, 2 * k + 2)
            defect_row_nodes.extend([triple] * 4)  # Hermite rows
            defect_row_nodes.extend([triple] * 4)  # Simpson rows
        blocks.append({
            "type": "eq",
            "fun": self.defects,
            "row_nodes": defect_row_nodes,
        })

        # Optional path constraints (e.g. obstacle): checked at every one of
        # the 2N+1 points, mesh and midpoints alike, so obstacles cannot be
        # cut through between nodes.
        for path_fn in p.path_constraints:
            def path_con(z, fn=path_fn):
                states, controls = self.unpack(z)
                return np.array([fn(states[i], controls[i]) for i in range(self.total_points)])

            blocks.append({
                "type": "ineq",
                "fun": path_con,
                "row_nodes": [(i,) for i in range(self.total_points)],
            })
        return blocks

    # ---------------------------------------------------------------- solving

    def solve(self, max_iter: int = 500, ftol: float = 1e-6, initial_guess=None) -> TrajectoryResult:
        """Runs the optimization.

        Args:
            max_iter: Maximum SLSQP iterations.
            ftol: SLSQP stopping tolerance on the objective.
            initial_guess: Optional (t, states, controls) from a previous solve (S5).
        """
        start_time = time.perf_counter()
        p = self.problem

        z0 = self._initial_guess(initial_guess)
        lo, hi = self._bounds()
        blocks = self.constraint_blocks()

        scipy_constraints = []
        for blk in blocks:
            con = {"type": blk["type"], "fun": blk["fun"]}
            if self.sparse_jacobian:
                # S3: build the pattern and column groups ONCE per block,
                # over the 2N+1-point variable layout (row_nodes index into
                # points, not mesh-only nodes).
                S = build_pattern(blk["row_nodes"], self.total_points, self.node_vars)
                groups = color_columns(S)
                con["jac"] = lambda z, f=blk["fun"], S=S, G=groups: sparse_fd_jacobian(f, z, S, G)
            scipy_constraints.append(con)

        grad = self.objective_grad if self.sparse_jacobian else None

        res = minimize(
            fun=self.objective,
            x0=z0,
            jac=grad,
            method="SLSQP",
            bounds=list(zip(lo, hi)),
            constraints=scipy_constraints,
            options={"maxiter": max_iter, "ftol": ftol, "disp": False},
        )

        solve_time = time.perf_counter() - start_time
        final_states, final_controls = self.unpack(res.x)

        enabled = [name for name, on in [
            ("S1", self.fix_endpoints_in_bounds), ("S2", self.vectorized),
            ("S3", self.sparse_jacobian)] if on]

        return TrajectoryResult(
            method_name="Hermite-Simpson Collocation (Improved)",
            success=bool(res.success),
            message=str(res.message),
            time=np.linspace(0.0, p.duration, self.total_points),
            state=final_states,
            control=final_controls,
            cost=float(self.objective(res.x)),
            solve_time=solve_time,
            iterations=int(getattr(res, "nit", 0)),
            max_constraint_violation=self.max_violation(res.x),
            info={"status": res.status, "total_nodes": self.total_points, "improvements": enabled},
        )


# =============================================================================
# S5: Warm start
# =============================================================================
#
# Identical idea to solve_with_warm_start() in improved_trapezoidal.py: solve
# a coarse grid first, stretch its answer (including the extra midpoints of
# the new, finer grid) onto the next grid size, and repeat until we reach N.


def default_coarse_levels(n_target: int, smallest: int = 8) -> List[int]:
    """Coarse grids to solve first: halve N repeatedly while it stays >= `smallest`."""
    levels = []
    n = n_target // 2
    while n >= smallest:
        levels.insert(0, n)
        n //= 2
    return levels


def solve_with_warm_start(
    problem: TrajectoryProblem,
    coarse_levels: Optional[Sequence[int]] = None,
    max_iter: int = 500,
    ftol: float = 1e-6,
    **solver_kwargs,
) -> TrajectoryResult:
    """S5: solve on coarse grids first, each one warm-starting the next.

    Args:
        problem: The problem at the final (fine) resolution.
        coarse_levels: Grid sizes to solve first, e.g. [10, 20].
                       Default: `default_coarse_levels(problem.n_nodes)`.
        max_iter, ftol: Passed to every solve.
        **solver_kwargs: S1-S3 flags passed to ImprovedHermiteSimpsonSolver.

    Returns:
        The result on the final grid. `solve_time` and `iterations` are the
        TOTALS over all levels (coarse solves included), for a fair
        comparison against a single direct solve.
    """
    if coarse_levels is None:
        coarse_levels = default_coarse_levels(problem.n_nodes)

    guess = None
    total_time, total_iters = 0.0, 0
    per_level = []
    for n in list(coarse_levels) + [problem.n_nodes]:
        sub_problem = dataclasses.replace(problem, n_nodes=n)
        res = ImprovedHermiteSimpsonSolver(sub_problem, **solver_kwargs).solve(
            max_iter=max_iter, ftol=ftol, initial_guess=guess
        )
        total_time += res.solve_time
        total_iters += res.iterations
        per_level.append({"N": n, "success": res.success, "iterations": res.iterations,
                          "time": res.solve_time})
        guess = (res.time, res.state, res.control)

    res.method_name = "Hermite-Simpson Collocation (Improved + Warm Start)"
    res.solve_time = total_time
    res.iterations = total_iters
    res.info["final_level_iterations"] = per_level[-1]["iterations"]
    res.info["levels"] = per_level
    res.info["improvements"] = res.info.get("improvements", []) + ["S5"]
    return res
