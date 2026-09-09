"""Physical parameters and configuration for the planar two-link manipulator."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ArmParameters:
    """Rigid body physical parameters for a 2-link planar arm.

    Attributes:
        l1: Length of link 1 [m].
        l2: Length of link 2 [m].
        m1: Mass of link 1 [kg].
        m2: Mass of link 2 [kg].
        r1: Distance from joint 1 to link 1 center of mass [m].
        r2: Distance from joint 2 to link 2 center of mass [m].
        I1: Rotational inertia of link 1 about its center of mass [kg*m^2].
        I2: Rotational inertia of link 2 about its center of mass [kg*m^2].
        g: Gravitational acceleration [m/s^2].
        tau_max: Actuator symmetric torque limit [N*m].
        q_min: Lower joint position limit [rad].
        q_max: Upper joint position limit [rad].
        dq_max: Maximum joint velocity limit [rad/s].
    """

    l1: float = 1.0
    l2: float = 1.0
    m1: float = 1.0
    m2: float = 1.0
    r1: float = 0.5
    r2: float = 0.5
    I1: float = (1.0 / 12.0) * 1.0 * (1.0**2)
    I2: float = (1.0 / 12.0) * 1.0 * (1.0**2)
    g: float = 9.81
    tau_max: float = 25.0
    q_min: float = -2.0 * np.pi
    q_max: float = 2.0 * np.pi
    dq_max: float = 15.0


DEFAULT_PARAMS = ArmParameters()

