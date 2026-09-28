from typing import Tuple
import numpy as np
from arm_opt.dynamics.parameters import ArmParameters, DEFAULT_PARAMS


def forward_kinematics(
    q: np.ndarray, params: ArmParameters = DEFAULT_PARAMS
) -> Tuple[np.ndarray, np.ndarray]:
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
    theta1 = q[0]
    theta12 = q[0] + q[1]
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s12, c12 = np.sin(theta12), np.cos(theta12)
    j11 = -params.l1 * s1 - params.l2 * s12
    j12 = -params.l2 * s12
    j21 = params.l1 * c1 + params.l2 * c12
    j22 = params.l2 * c12
    return np.array([[j11, j12], [j21, j22]], dtype=float)
