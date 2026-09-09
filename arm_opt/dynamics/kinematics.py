"""Forward kinematics and differential mappings for the planar two-link manipulator."""

from typing import Tuple
import numpy as np
from arm_opt.dynamics.parameters import ArmParameters, DEFAULT_PARAMS


def forward_kinematics(
    q: np.ndarray, params: ArmParameters = DEFAULT_PARAMS
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes Cartesian positions of elbow joint and end-effector.

    Args:
        q: Joint angles [theta1, theta2] in radians.
        params: Physical arm parameters.

    Returns:
        p_elbow: [x1, y1] coordinates of elbow joint.
        p_ee: [x2, y2] coordinates of end-effector.
    """
    theta1 = q[0]
    theta12 = q[0] + q[1]

    x1 = params.l1 * np.cos(theta1)
    y1 = params.l1 * np.sin(theta1)

    x2 = x1 + params.l2 * np.cos(theta12)
    y2 = y1 + params.l2 * np.sin(theta12)

    return np.array([x1, y1], dtype=float), np.array([x2, y2], dtype=float)


def end_effector_jacobian(
    q: np.ndarray, params: ArmParameters = DEFAULT_PARAMS
) -> np.ndarray:
    """Computes the 2x2 analytical Jacobian J(q) = d p_ee / d q."""
    theta1 = q[0]
    theta12 = q[0] + q[1]

    s1 = np.sin(theta1)
    c1 = np.cos(theta1)
    s12 = np.sin(theta12)
    c12 = np.cos(theta12)

    j11 = -params.l1 * s1 - params.l2 * s12
    j12 = -params.l2 * s12

    j21 = params.l1 * c1 + params.l2 * c12
    j22 = params.l2 * c12

    return np.array([[j11, j12], [j21, j22]], dtype=float)

