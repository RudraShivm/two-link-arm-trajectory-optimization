from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ArmParameters:
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
