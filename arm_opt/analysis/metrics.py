from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from arm_opt.dynamics.kinematics import forward_kinematics
from arm_opt.solvers.base import TrajectoryResult, TrajectoryProblem


@dataclass
class TrajectoryMetrics:
    method_name: str
    success: bool
    solve_time: float
    iterations: int
    cost: float
    total_effort: float
    peak_torque: float
    terminal_state_error: float
    terminal_ee_error_meters: float
    max_constraint_violation: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method_name,
            "success": self.success,
            "solve_time_sec": round(self.solve_time, 4),
            "iterations": self.iterations,
            "cost": round(self.cost, 4),
            "total_effort": round(self.total_effort, 4),
            "peak_torque_Nm": round(self.peak_torque, 3),
            "terminal_state_err": round(self.terminal_state_error, 5),
            "terminal_ee_err_m": round(self.terminal_ee_error_meters, 5),
            "max_violation": round(self.max_constraint_violation, 6),
        }


def compute_metrics(
    problem: TrajectoryProblem, result: TrajectoryResult
) -> TrajectoryMetrics:
    t = result.time
    q = result.state[:, :2]
    u = result.control

    dt = np.diff(t)
    u_norm_sq = np.sum(u**2, axis=1)
    total_effort = float(np.sum(0.5 * dt * (u_norm_sq[:-1] + u_norm_sq[1:])))
    peak_torque = float(np.max(np.abs(u)))
    term_state_err = float(np.linalg.norm(result.state[-1] - problem.xf))

    _, ee_final = forward_kinematics(q[-1], problem.arm_params)
    _, ee_target = forward_kinematics(problem.xf[:2], problem.arm_params)
    terminal_ee_err = float(np.linalg.norm(ee_final - ee_target))

    return TrajectoryMetrics(
        method_name=result.method_name,
        success=result.success,
        solve_time=result.solve_time,
        iterations=result.iterations,
        cost=result.cost,
        total_effort=total_effort,
        peak_torque=peak_torque,
        terminal_state_error=term_state_err,
        terminal_ee_error_meters=terminal_ee_err,
        max_constraint_violation=result.max_constraint_violation,
    )
