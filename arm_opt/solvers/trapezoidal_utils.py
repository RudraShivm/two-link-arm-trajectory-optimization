"""Accuracy helpers for trapezoidal collocation (A1-A4).

Trapezoidal collocation assumes the state derivative f changes LINEARLY between
nodes. Integrating a linear f gives a QUADRATIC state, so the state between
nodes is a parabola, not a straight line (Kelly 2017, Sec. 4).
"""

from typing import Tuple

import numpy as np

from arm_opt.dynamics.manipulator import TwoLinkArm
from arm_opt.solvers.improved_trapezoidal import state_derivative_batch


def trapezoidal_interpolate(
    arm: TwoLinkArm,
    t_nodes: np.ndarray,
    states: np.ndarray,
    controls: np.ndarray,
    t_query,
) -> Tuple[np.ndarray, np.ndarray]:
    """A1: state (quadratic) and control (linear) between nodes.

    For t = t_k + tau with 0 <= tau <= h_k:
        x(t) = x_k + f_k * tau + (tau^2 / (2 h_k)) * (f_{k+1} - f_k)
        u(t) = u_k + (tau / h_k) * (u_{k+1} - u_k)

    At tau = h_k this gives x_k + (h_k/2)(f_k + f_{k+1}), which equals x_{k+1}
    exactly when the defect is zero. Works with non-uniform grids too.

    Args:
        arm: Dynamics model, used for f(x, u).
        t_nodes: Node times, shape (N+1,).
        states: Node states, shape (N+1, 4).
        controls: Node controls, shape (N+1, 2).
        t_query: Times to evaluate at (scalar or array of shape (M,)).

    Returns:
        (x, u) with shapes (M, 4) and (M, 2).
    """
    t_nodes = np.asarray(t_nodes, dtype=float)
    t_query = np.atleast_1d(np.asarray(t_query, dtype=float))
    f = state_derivative_batch(arm.p, states, controls)  # (N+1, 4)

    # Interval index k with t_k <= t < t_{k+1}; the last node belongs to the last interval.
    k = np.clip(np.searchsorted(t_nodes, t_query, side="right") - 1, 0, len(t_nodes) - 2)
    h = (t_nodes[k + 1] - t_nodes[k])[:, None]
    tau = (t_query - t_nodes[k])[:, None]

    x = states[k] + f[k] * tau + (tau**2 / (2.0 * h)) * (f[k + 1] - f[k])
    u = controls[k] + (tau / h) * (controls[k + 1] - controls[k])
    return x, u


def make_state_interp_fn(problem, result):
    """A1: t -> quadratic state, ready to pass as `perform_reality_check(..., state_interp_fn=...)`."""

    def state_interp_fn(t):
        x, _ = trapezoidal_interpolate(problem.arm, result.time, result.state, result.control, t)
        return x

    return state_interp_fn


def fit_order(hs, errors) -> float:
    """A2: slope of log(error) vs log(h), i.e. the observed order p in error ≈ C·h^p."""
    slope, _ = np.polyfit(np.log(np.asarray(hs, dtype=float)), np.log(np.asarray(errors, dtype=float)), 1)
    return float(slope)


def point_segment_distance(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Distance from point c to segments a[i]-b[i]. a, b: (M, 2), c: (2,) -> (M,)."""
    ab = b - a
    s = np.clip(np.sum((c - a) * ab, axis=1) / np.sum(ab * ab, axis=1), 0.0, 1.0)
    closest = a + s[:, None] * ab
    return np.linalg.norm(closest - c, axis=1)


def link_clearances(params, angles: np.ndarray, center, radius: float):
    """A3: clearance (distance minus radius, metres; < 0 = collision) of each part of the arm.

    Args:
        params: ArmParameters (link lengths).
        angles: Joint angles, shape (M, 2) (or states of shape (M, 4)).
        center, radius: The circular obstacle.

    Returns:
        dict with arrays of shape (M,): "link1", "link2", "hand".
    """
    q = np.atleast_2d(angles)[:, :2]
    center = np.asarray(center, dtype=float)
    elbow = np.column_stack([params.l1 * np.cos(q[:, 0]), params.l1 * np.sin(q[:, 0])])
    hand = elbow + np.column_stack([params.l2 * np.cos(q[:, 0] + q[:, 1]), params.l2 * np.sin(q[:, 0] + q[:, 1])])
    shoulder = np.zeros_like(elbow)
    return {
        "link1": point_segment_distance(shoulder, elbow, center) - radius,
        "link2": point_segment_distance(elbow, hand, center) - radius,
        "hand": np.linalg.norm(hand - center, axis=1) - radius,
    }


def path_clearance(problem, result, center, radius: float, samples_per_interval: int = 20):
    """A4: clearances along the whole trajectory, sampling the A1 parabola inside every interval.

    Returns:
        (t, clearances) where clearances is the dict from `link_clearances`, sampled at
        `samples_per_interval` points per interval (nodes included).
    """
    t_nodes = np.asarray(result.time, dtype=float)
    frac = np.arange(samples_per_interval) / samples_per_interval
    t = np.concatenate([t_nodes[:-1, None] + frac[None, :] * np.diff(t_nodes)[:, None]]).ravel()
    t = np.append(t, t_nodes[-1])
    x, _ = trapezoidal_interpolate(problem.arm, t_nodes, result.state, result.control, t)
    return t, link_clearances(problem.arm_params, x, center, radius)
