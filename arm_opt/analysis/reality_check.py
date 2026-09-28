from dataclasses import dataclass
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from arm_opt.dynamics.kinematics import forward_kinematics
from arm_opt.solvers.base import TrajectoryProblem, TrajectoryResult


@dataclass
class RealityCheckResult:
    t_sim: np.ndarray
    x_sim: np.ndarray
    x_opt_interp: np.ndarray
    max_state_drift: float
    max_cartesian_drift: float
    terminal_cartesian_drift: float


def perform_reality_check(
    problem: TrajectoryProblem,
    result: TrajectoryResult,
    num_eval_points: int = 200,
) -> RealityCheckResult:
    """Replay optimal u(t) with DOP853 and compare to the optimizer state trajectory."""
    t_opt = result.time
    u_opt = result.control
    x_opt = result.state
    arm = problem.arm
    params = problem.arm_params

    control_interpolator = interp1d(
        t_opt, u_opt, axis=0, kind="linear", fill_value="extrapolate"
    )

    def continuous_dynamics(t: float, x: np.ndarray) -> np.ndarray:
        return arm.state_derivative(x, control_interpolator(t))

    t_eval = np.linspace(0.0, float(problem.duration), num_eval_points)
    sim_sol = solve_ivp(
        fun=continuous_dynamics,
        t_span=(0.0, float(problem.duration)),
        y0=problem.x0,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-8,
    )

    t_sim = sim_sol.t
    x_sim = sim_sol.y.T
    x_opt_interp = interp1d(t_opt, x_opt, axis=0, kind="linear")(t_sim)

    max_state_drift = float(np.max(np.linalg.norm(x_sim - x_opt_interp, axis=1)))

    ee_drifts = np.zeros(len(t_sim))
    for i in range(len(t_sim)):
        _, p_sim = forward_kinematics(x_sim[i, :2], params)
        _, p_opt = forward_kinematics(x_opt_interp[i, :2], params)
        ee_drifts[i] = np.linalg.norm(p_sim - p_opt)

    _, p_sim_final = forward_kinematics(x_sim[-1, :2], params)
    _, p_target = forward_kinematics(problem.xf[:2], params)

    return RealityCheckResult(
        t_sim=t_sim,
        x_sim=x_sim,
        x_opt_interp=x_opt_interp,
        max_state_drift=max_state_drift,
        max_cartesian_drift=float(np.max(ee_drifts)),
        terminal_cartesian_drift=float(np.linalg.norm(p_sim_final - p_target)),
    )
