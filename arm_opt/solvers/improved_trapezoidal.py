"""Method 2 (Improved): Trapezoidal Direct Collocation with speed improvements S1-S5.

This file is a faster version of `trapezoidal.py`. The MATH PROBLEM IS THE SAME
(same unknowns, same effort objective, same trapezoidal defects), so the answer
should be the same too. Only HOW we hand the problem to the optimizer changes.

The original `trapezoidal.py` is left untouched so it can serve as the baseline
for before/after comparisons.

Improvements (each one can be switched on/off with a flag, for comparisons):

    S1  fix_endpoints_in_bounds : lock start/end states with bounds (min = max)
                                  instead of 8 equality constraints.
    S2  vectorized              : compute the physics f(x, u) for ALL nodes in
                                  one numpy call instead of a Python loop.
    S3  sparse_jacobian         : give the optimizer derivatives:
                                  - exact gradient of the effort objective
                                  - constraint Jacobian via "colored" finite
                                    differences (12 evaluations instead of 186)
    S4  scaling                 : let the optimizer work with variables of size ~1
                                  (velocity / 15, torque / 30), and divide the
                                  objective by h * tau_max^2 to match.
    S5  warm start              : `solve_with_warm_start()` solves a coarse grid
                                  first and uses that answer as the initial guess.

Quick reminder of the problem (see docs/TRAPEZOIDAL_IMPROVEMENT_PLAN.md):
    unknowns  z = [x_0, u_0, x_1, u_1, ..., x_N, u_N]   (6 numbers per node)
    minimize  effort = trapezoid-integral of (tau^2 + 0.001 * dq^2)
    such that x_0 = A, x_N = B,
              x_{k+1} - x_k - (h/2)(f_k + f_{k+1}) = 0   for every interval k
              bounds on every variable, optional obstacle constraints >= 0
"""

import dataclasses
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from arm_opt.dynamics.parameters import ArmParameters
from arm_opt.solvers.base import TrajectoryProblem, TrajectoryResult


# =============================================================================
# S2: Vectorized physics
# =============================================================================
#
# The original code calls `arm.state_derivative(x, u)` once per node inside a
# Python `for` loop. Python loops are slow. Here we compute the SAME formulas,
# but every variable is an array holding the value for ALL nodes at once, so
# numpy does the work in fast C code.
#
# The formulas are copied from arm_opt/dynamics/manipulator.py (mass matrix,
# Coriolis, gravity). We keep this function here instead of editing
# manipulator.py, because manipulator.py is shared with the other solvers.


def state_derivative_batch(params: ArmParameters, X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Computes f(x, u) = [dq, ddq] for many nodes at once.

    Args:
        params: Physical arm parameters (lengths, masses, ...).
        X: States of shape (K, 4): each row is [theta1, theta2, dtheta1, dtheta2].
        U: Controls of shape (K, 2): each row is [tau1, tau2].

    Returns:
        Array of shape (K, 4): each row is [dtheta1, dtheta2, ddtheta1, ddtheta2].
    """
    p = params
    # Columns of X. Each of these is an array of length K (one value per node).
    q1, q2, dq1, dq2 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    c2, s2 = np.cos(q2), np.sin(q2)

    # Mass matrix M(q) = [[m11, m12], [m12, m22]]  (same as TwoLinkArm.mass_matrix)
    m11 = p.m1 * p.r1**2 + p.I1 + p.m2 * (p.l1**2 + p.r2**2 + 2.0 * p.l1 * p.r2 * c2) + p.I2
    m12 = p.m2 * (p.r2**2 + p.l1 * p.r2 * c2) + p.I2
    m22 = p.m2 * p.r2**2 + p.I2  # a constant (does not depend on q)

    # Coriolis / centrifugal vector  (same as TwoLinkArm.coriolis_vector)
    h = p.m2 * p.l1 * p.r2 * s2
    cor1 = -h * (2.0 * dq1 * dq2 + dq2**2)
    cor2 = h * dq1**2

    # Gravity vector  (same as TwoLinkArm.gravity_vector)
    c1, c12 = np.cos(q1), np.cos(q1 + q2)
    g1 = (p.m1 * p.r1 + p.m2 * p.l1) * p.g * c1 + p.m2 * p.r2 * p.g * c12
    g2 = p.m2 * p.r2 * p.g * c12

    # Newton's law: M * ddq = tau - C*dq - g  ->  solve for ddq.
    # For a 2x2 matrix the inverse has a simple closed form:
    #   [[a, b], [b, d]]^-1 = 1/(a*d - b^2) * [[d, -b], [-b, a]]
    b1 = U[:, 0] - cor1 - g1
    b2 = U[:, 1] - cor2 - g2
    det = m11 * m22 - m12**2
    ddq1 = (m22 * b1 - m12 * b2) / det
    ddq2 = (-m12 * b1 + m11 * b2) / det

    return np.column_stack([dq1, dq2, ddq1, ddq2])


# =============================================================================
# S3: Sparse ("colored") finite-difference Jacobian helpers
# =============================================================================
#
# The optimizer needs to know how every constraint changes when every variable
# changes (the Jacobian, a big table: rows = constraints, columns = variables).
#
# Without help, scipy fills this table by nudging ONE variable at a time and
# re-evaluating ALL constraints: 186 evaluations per iteration for N = 30.
#
# Key observation: defect k only involves nodes k and k+1. So nudging node 3
# and node 7 AT THE SAME TIME is fine: no single defect sees both nudges, so we
# can still tell which change came from which node.
#
# Classroom analogy: students in a row only hear their left/right neighbour.
# Tell something to all odd-numbered students at once, then all even ones.
# Two rounds instead of one round per student.
#
# The helpers below do this for ANY constraint as long as we know, for each row,
# which nodes it depends on ("row_nodes"). That way obstacle constraints (and
# Arpa's future midpoint constraints) automatically get fast Jacobians too.


def build_pattern(row_nodes: Sequence[Tuple[int, ...]], n_nodes: int, node_vars: int = 6) -> np.ndarray:
    """Builds the sparsity pattern: which variables each constraint row may depend on.

    Args:
        row_nodes: For every constraint row r, a tuple of node indices that row uses.
                   Example: defect k's rows use nodes (k, k+1).
        n_nodes: Total number of nodes (N + 1).
        node_vars: Variables per node (4 states + 2 controls = 6).

    Returns:
        Boolean matrix S of shape (rows, n_nodes * node_vars).
        S[r, c] = True means "row r may depend on variable c".
        (Marking a few extra True entries is safe, just slightly slower.
         Missing a True entry would give WRONG derivatives.)
    """
    S = np.zeros((len(row_nodes), n_nodes * node_vars), dtype=bool)
    for r, nodes in enumerate(row_nodes):
        for k in nodes:
            # All 6 variables of node k (columns k*6 ... k*6+5).
            S[r, k * node_vars : (k + 1) * node_vars] = True
    return S


def color_columns(S: np.ndarray) -> List[np.ndarray]:
    """Groups columns (variables) so that no two columns in a group share a row.

    Columns in the same group can be nudged together in ONE evaluation.
    This is a simple "greedy" method: go through the columns in order and put
    each one into the first group where it does not clash with anyone.

    For the defect constraints this produces 12 groups
    (6 variables x {even nodes, odd nodes}).

    Returns:
        A list of arrays; each array holds the column indices of one group.
    """
    groups = []  # each entry: [list_of_columns, boolean mask of rows already used]
    for col in range(S.shape[1]):
        rows = S[:, col]
        for g in groups:
            if not np.any(g[1] & rows):  # no shared row -> no clash
                g[0].append(col)
                g[1] |= rows
                break
        else:  # (for-else: runs only if we never hit `break`) -> start a new group
            groups.append([[col], rows.copy()])
    return [np.array(g[0]) for g in groups]


def sparse_fd_jacobian(
    fun: Callable[[np.ndarray], np.ndarray],
    z: np.ndarray,
    S: np.ndarray,
    groups: List[np.ndarray],
    rel_step: float = 1e-7,
) -> np.ndarray:
    """Finite-difference Jacobian using one extra evaluation per column group.

    Plain finite differences:  dF/dz_c ~ (F(z + step * e_c) - F(z)) / step
    Here we nudge ALL columns of a group together. Because (by construction) no
    row depends on two columns of the same group, each row's change can be
    blamed on exactly one column.

    Args:
        fun: Constraint function z -> array of shape (rows,).
        z: Point where we want the Jacobian.
        S: Sparsity pattern from `build_pattern`.
        groups: Column groups from `color_columns`.
        rel_step: Relative nudge size.

    Returns:
        Dense Jacobian of shape (rows, len(z)) (mostly zeros).
    """
    f0 = fun(z)
    J = np.zeros((f0.size, z.size))
    # Nudge size per variable: bigger variables get bigger nudges.
    steps = rel_step * np.maximum(1.0, np.abs(z))
    for cols in groups:
        z_pert = z.copy()
        z_pert[cols] += steps[cols]  # nudge the whole group at once
        df = fun(z_pert) - f0
        for c in cols:
            rows = S[:, c]  # the rows that column c can affect
            J[rows, c] = df[rows] / steps[c]
    return J


# =============================================================================
# The improved solver
# =============================================================================


class ImprovedTrapezoidalSolver:
    """Trapezoidal Direct Collocation with speed improvements S1-S4.

    Same decision variables, objective and defects as TrapezoidalCollocationSolver.
    All improvements are ON by default. Turn them off one by one to measure how
    much each one helps (see experiments/compare_trapezoidal.py).

    For S5 (warm start) use the function `solve_with_warm_start()` below.
    """

    def __init__(
        self,
        problem: TrajectoryProblem,
        fix_endpoints_in_bounds: bool = True,  # S1
        vectorized: bool = True,  # S2
        sparse_jacobian: bool = True,  # S3
        scaling: bool = True,  # S4
        check_midpoints: bool = False,  # A4
    ):
        self.problem = problem
        self.arm = problem.arm
        self.N = problem.n_nodes  # number of intervals (so N + 1 nodes)
        self.dt = problem.dt  # h = T / N
        self.n_states = 4
        self.n_controls = 2
        self.node_vars = self.n_states + self.n_controls  # 6
        self.n_vars = (self.N + 1) * self.node_vars

        self.fix_endpoints_in_bounds = fix_endpoints_in_bounds
        self.vectorized = vectorized
        self.sparse_jacobian = sparse_jacobian
        self.scaling = scaling
        self.check_midpoints = check_midpoints

    # ------------------------------------------------------------------ helpers

    def unpack(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Flat vector z (186,) -> states (N+1, 4) and controls (N+1, 2)."""
        Z = z.reshape((self.N + 1, self.node_vars))
        return Z[:, : self.n_states], Z[:, self.n_states :]

    def pack(self, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """States (N+1, 4) and controls (N+1, 2) -> flat vector z (186,)."""
        return np.hstack([states, controls]).flatten()

    def dynamics_at_nodes(self, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """f(x_k, u_k) for every node k, shape (N+1, 4)."""
        if self.vectorized:
            # S2: one numpy call for all nodes
            return state_derivative_batch(self.problem.arm_params, states, controls)
        # Original behaviour: Python loop, one node at a time
        return np.array(
            [self.arm.state_derivative(states[k], controls[k]) for k in range(self.N + 1)]
        )

    def defects(self, z: np.ndarray) -> np.ndarray:
        """All trapezoidal defects, flattened to shape (4N,).

        defect_k = x_{k+1} - x_k - (h/2) * (f_k + f_{k+1})   for k = 0 .. N-1

        Instead of looping over k, we use array slices:
            states[1:]  = x_1 ... x_N      ("next" node of every interval)
            states[:-1] = x_0 ... x_{N-1}  ("current" node of every interval)
        """
        states, controls = self.unpack(z)
        f = self.dynamics_at_nodes(states, controls)
        d = states[1:] - states[:-1] - 0.5 * self.dt * (f[:-1] + f[1:])
        return d.flatten()  # row order: defect 0 (4 rows), defect 1 (4 rows), ...

    def midpoints(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """A4: state and control at the middle of every interval, shape (N, 4) and (N, 2).

        Trapezoidal collocation makes the state a parabola inside each interval (A1).
        At tau = h/2 that parabola simplifies to
            x_mid = x_k + (h/8) * (3 f_k + f_{k+1})
            u_mid = (u_k + u_{k+1}) / 2
        """
        states, controls = self.unpack(z)
        f = self.dynamics_at_nodes(states, controls)
        x_mid = states[:-1] + (self.dt / 8.0) * (3.0 * f[:-1] + f[1:])
        u_mid = 0.5 * (controls[:-1] + controls[1:])
        return x_mid, u_mid

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

        - No guess given: same as the original solver (straight line from A to B
          for the states, "just hold the arm against gravity" for the torques).
        - Guess given (S5, warm start): a previous solution (t_old, X_old, U_old),
          possibly on a different grid, is stretched onto our grid by linear
          interpolation, one column at a time.
        """
        p = self.problem
        if initial_guess is None:
            states = p.get_linear_state_guess()
            controls = np.array(
                [np.clip(self.arm.gravity_vector(states[k, :2]), -p.tau_max, p.tau_max)
                 for k in range(self.N + 1)]
            )
        else:
            t_old, X_old, U_old = initial_guess
            t_new = p.time_grid
            states = np.column_stack([np.interp(t_new, t_old, X_old[:, i]) for i in range(4)])
            controls = np.column_stack([np.interp(t_new, t_old, U_old[:, i]) for i in range(2)])
            # Endpoints must match exactly (important when S1 locks them).
            states[0], states[-1] = p.x0, p.xf
        return self.pack(states, controls)

    def _bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Lower and upper bound for every variable, as two arrays of length n_vars."""
        p = self.problem
        # Bounds for ONE node, in the order [q1, q2, dq1, dq2, tau1, tau2]
        node_lo = np.array([p.q_min, p.q_min, -p.dq_max, -p.dq_max, -p.tau_max, -p.tau_max])
        node_hi = np.array([p.q_max, p.q_max, p.dq_max, p.dq_max, p.tau_max, p.tau_max])
        # Repeat them for all N+1 nodes, as (N+1, 6) tables
        lo = np.tile(node_lo, (self.N + 1, 1))
        hi = np.tile(node_hi, (self.N + 1, 1))

        if self.fix_endpoints_in_bounds:
            # S1: lock the 4 states of the first and last node.
            # min = max = required value -> the optimizer cannot move them at all,
            # so we no longer need the 8 equations "x_0 - A = 0", "x_N - B = 0".
            lo[0, :4] = hi[0, :4] = p.x0
            lo[-1, :4] = hi[-1, :4] = p.xf
        return lo.flatten(), hi.flatten()

    def objective(self, z: np.ndarray) -> float:
        """Effort = trapezoid-integral of (tau1^2 + tau2^2 + c*(dq1^2 + dq2^2)).

        Identical to the original solver.
        Composite trapezoid: (h/2) * [L_0 + 2 L_1 + ... + 2 L_{N-1} + L_N].
        """
        states, controls = self.unpack(z)
        L = np.sum(controls**2, axis=1) + self.problem.effort_weight_dq * np.sum(states[:, 2:] ** 2, axis=1)
        return float(0.5 * self.dt * (L[0] + 2.0 * np.sum(L[1:-1]) + L[-1]))

    def objective_grad(self, z: np.ndarray) -> np.ndarray:
        """S3(a): EXACT gradient of the objective (no finite differences needed).

        Write the cost as  cost = (h/2) * sum_k w_k * L_k  with weights
        w = [1, 2, 2, ..., 2, 1]. Then, by ordinary differentiation:
            d cost / d tau_k   = (h/2) * w_k * 2 * tau_k       = h * w_k * tau_k
            d cost / d dq_k    = (h/2) * w_k * 2 * c * dq_k    = h * w_k * c * dq_k
            d cost / d theta_k = 0   (angles do not appear in the cost)
        """
        states, controls = self.unpack(z)
        h = self.dt
        w = np.full(self.N + 1, 2.0)
        w[0] = w[-1] = 1.0
        G = np.zeros((self.N + 1, self.node_vars))  # same layout as the (N+1, 6) table
        G[:, 2:4] = h * w[:, None] * self.problem.effort_weight_dq * states[:, 2:4]
        G[:, 4:6] = h * w[:, None] * controls
        return G.flatten()

    def constraint_blocks(self) -> List[Dict]:
        """All constraints, as a list of "blocks".

        Each block is a dict:
            "type":      "eq" (must be = 0) or "ineq" (must be >= 0)
            "fun":       function z -> array of numbers
            "row_nodes": for every row of "fun", the nodes that row depends on
                         (needed by the sparse Jacobian in S3)
        """
        p = self.problem
        N = self.N
        blocks = []

        if not self.fix_endpoints_in_bounds:
            # Without S1: start/end conditions are equality constraints (like the original).
            blocks.append({
                "type": "eq",
                "fun": lambda z: self.unpack(z)[0][0] - p.x0,
                "row_nodes": [(0,)] * 4,  # 4 rows, all depend only on node 0
            })
            blocks.append({
                "type": "eq",
                "fun": lambda z: self.unpack(z)[0][-1] - p.xf,
                "row_nodes": [(N,)] * 4,  # 4 rows, all depend only on node N
            })

        # Defects: 4 rows per interval k, each depends on nodes k and k+1.
        blocks.append({
            "type": "eq",
            "fun": self.defects,
            "row_nodes": [(k, k + 1) for k in range(N) for _ in range(4)],
        })

        # Optional path constraints (e.g. obstacle): one row per node, >= 0.
        for path_fn in p.path_constraints:
            # `fn=path_fn` freezes the current function inside the loop
            # (same Python trick as `make_path_con` in the original file).
            def path_con(z, fn=path_fn):
                states, controls = self.unpack(z)
                return np.array([fn(states[k], controls[k]) for k in range(N + 1)])

            blocks.append({
                "type": "ineq",
                "fun": path_con,
                "row_nodes": [(k,) for k in range(N + 1)],  # row k uses only node k
            })

            if self.check_midpoints:
                # A4: the same constraint in the middle of every interval, so the
                # path cannot cut through an obstacle between two nodes.
                def midpoint_con(z, fn=path_fn):
                    x_mid, u_mid = self.midpoints(z)
                    return np.array([fn(x_mid[k], u_mid[k]) for k in range(N)])

                blocks.append({
                    "type": "ineq",
                    "fun": midpoint_con,
                    "row_nodes": [(k, k + 1) for k in range(N)],  # midpoint k uses nodes k and k+1
                })
        return blocks

    # ---------------------------------------------------------------- solving

    def solve(self, max_iter: int = 400, ftol: float = 1e-6, initial_guess=None) -> TrajectoryResult:
        """Runs the optimization.

        Args:
            max_iter: Maximum SLSQP iterations.
            ftol: SLSQP stopping tolerance on the objective.
            initial_guess: Optional (t, states, controls) from a previous solve (S5).
        """
        start_time = time.perf_counter()  # everything below counts as solve time

        z0 = self._initial_guess(initial_guess)
        lo, hi = self._bounds()
        blocks = self.constraint_blocks()

        # ---- S4: scaling -----------------------------------------------------
        # The optimizer works with y = z / s instead of z, where s is a
        # "typical size" for each variable:
        #   angles  -> 1        (already ~1)
        #   dq      -> dq_max   (~15, so dq/15 is between -1 and 1)
        #   tau     -> tau_max  (~30, so tau/30 is between -1 and 1)
        # Then every variable the optimizer sees is roughly between -1 and 1.
        # Whenever our functions are called we convert back: z = y * s.
        #
        # We ALSO divide the objective by obj_scale = h * tau_max^2. Why?
        # SLSQP starts by assuming the objective's "curvature" (2nd derivative)
        # is about 1 in every direction. After dividing tau by 30, the curvature
        # in the scaled-torque direction becomes 900x bigger (~ 2*h*w*tau_max^2),
        # which is far from 1, so SLSQP needs many extra iterations to learn the
        # true shape. Dividing the cost by h*tau_max^2 brings that curvature back
        # to ~2-4. We measured this: scaling only the variables made rest_to_rest
        # (N=40) go from 46 to 115 iterations; scaling both keeps it at 46 and
        # cuts high_speed from 111 to 56 iterations.
        p = self.problem
        if self.scaling:
            s = np.tile([1.0, 1.0, p.dq_max, p.dq_max, p.tau_max, p.tau_max], self.N + 1)
            obj_scale = self.dt * p.tau_max**2
        else:
            s = np.ones(self.n_vars)  # no scaling: y = z
            obj_scale = 1.0

        # Objective in y-space.
        def fun_y(y):
            return self.objective(y * s) / obj_scale

        # Chain rule: d cost/dy = (d cost/dz) * (dz/dy) = grad_z * s
        grad_y = (lambda y: self.objective_grad(y * s) * s / obj_scale) if self.sparse_jacobian else None
        # SLSQP's ftol is compared against changes in the (scaled) objective, so
        # scale it the same way. Then "stop" means the same thing as before.
        ftol = ftol / obj_scale

        scipy_constraints = []
        for blk in blocks:
            con = {"type": blk["type"], "fun": lambda y, f=blk["fun"]: f(y * s)}
            if self.sparse_jacobian:
                # S3(b): build the pattern and column groups ONCE per block,
                # then every Jacobian call needs only len(groups) extra evaluations.
                S = build_pattern(blk["row_nodes"], self.N + 1, self.node_vars)
                groups = color_columns(S)
                # Jacobian in z-space, then chain rule: dF/dy = dF/dz * s (per column).
                con["jac"] = lambda y, f=blk["fun"], S=S, G=groups: (
                    sparse_fd_jacobian(f, y * s, S, G) * s[None, :]
                )
            scipy_constraints.append(con)

        res = minimize(
            fun=fun_y,
            x0=z0 / s,
            jac=grad_y,  # None -> scipy estimates the gradient itself (original behaviour)
            method="SLSQP",
            bounds=list(zip(lo / s, hi / s)),
            constraints=scipy_constraints,
            options={"maxiter": max_iter, "ftol": ftol, "disp": False},
        )

        solve_time = time.perf_counter() - start_time
        z_final = res.x * s  # back from y-space to real units
        final_states, final_controls = self.unpack(z_final)

        enabled = [name for name, on in [
            ("S1", self.fix_endpoints_in_bounds), ("S2", self.vectorized),
            ("S3", self.sparse_jacobian), ("S4", self.scaling),
            ("A4", self.check_midpoints)] if on]

        return TrajectoryResult(
            method_name="Trapezoidal Collocation (Improved)",
            success=bool(res.success),
            message=str(res.message),
            time=self.problem.time_grid,
            state=final_states,
            control=final_controls,
            cost=float(self.objective(z_final)),
            solve_time=solve_time,
            iterations=int(getattr(res, "nit", 0)),
            max_constraint_violation=self.max_violation(z_final),
            info={"status": res.status, "improvements": enabled},
        )


# =============================================================================
# S5: Warm start
# =============================================================================
#
# The default initial guess (straight line, zero velocity everywhere) is
# physically impossible, so the optimizer has a long way to travel.
# Idea: like sketching with a pencil before drawing details:
#   1. solve a small, cheap problem (e.g. N = 10)
#   2. stretch its answer onto a finer grid and use it as the starting guess
#   3. repeat until we reach the N we actually want
# Each solve starts close to its answer, so it needs few iterations.


def default_coarse_levels(n_target: int, smallest: int = 8) -> List[int]:
    """Coarse grids to solve first: halve N repeatedly while it stays >= `smallest`.

    Examples: 30 -> [15], 40 -> [10, 20], 80 -> [10, 20, 40], 10 -> [].
    """
    levels = []
    n = n_target // 2
    while n >= smallest:
        levels.insert(0, n)
        n //= 2
    return levels


def solve_with_warm_start(
    problem: TrajectoryProblem,
    coarse_levels: Optional[Sequence[int]] = None,
    max_iter: int = 400,
    ftol: float = 1e-6,
    **solver_kwargs,
) -> TrajectoryResult:
    """S5: solve on coarse grids first, each one warm-starting the next.

    Args:
        problem: The problem at the final (fine) resolution.
        coarse_levels: Grid sizes to solve first, e.g. [10, 20].
                       Default: `default_coarse_levels(problem.n_nodes)`.
        max_iter, ftol: Passed to every solve.
        **solver_kwargs: S1-S4 flags passed to ImprovedTrapezoidalSolver.

    Returns:
        The result on the final grid. For a FAIR comparison, `solve_time` and
        `iterations` are the TOTALS over all levels (coarse solves included).
    """
    if coarse_levels is None:
        coarse_levels = default_coarse_levels(problem.n_nodes)

    guess = None
    total_time, total_iters = 0.0, 0
    per_level = []
    for n in list(coarse_levels) + [problem.n_nodes]:
        # Same problem, different number of intervals.
        sub_problem = dataclasses.replace(problem, n_nodes=n)
        res = ImprovedTrapezoidalSolver(sub_problem, **solver_kwargs).solve(
            max_iter=max_iter, ftol=ftol, initial_guess=guess
        )
        total_time += res.solve_time
        total_iters += res.iterations
        per_level.append({"N": n, "success": res.success, "iterations": res.iterations,
                          "time": res.solve_time})
        guess = (res.time, res.state, res.control)  # becomes the next level's guess

    res.method_name = "Trapezoidal Collocation (Improved + Warm Start)"
    res.solve_time = total_time
    res.iterations = total_iters
    res.info["final_level_iterations"] = per_level[-1]["iterations"]
    res.info["levels"] = per_level
    res.info["improvements"] = res.info.get("improvements", []) + ["S5"]
    return res
