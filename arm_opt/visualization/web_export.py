import json
from typing import Dict, Optional, Tuple, Any
import numpy as np
from arm_opt.dynamics.kinematics import forward_kinematics
from arm_opt.solvers.base import TrajectoryProblem, TrajectoryResult
from arm_opt.analysis.reality_check import perform_reality_check
from arm_opt.analysis.metrics import compute_metrics


def build_scenario_dict(
    problem: TrajectoryProblem,
    results: Dict[str, TrajectoryResult],
    obstacle_spec: Optional[Tuple[Tuple[float, float], float]] = None,
    scenario_name: str = "Trajectory Optimization",
    scenario_desc: str = "",
) -> Dict[str, Any]:
    arm = problem.arm
    params = problem.arm_params
    _, target_ee = forward_kinematics(problem.xf[:2], params)
    _, initial_ee = forward_kinematics(problem.x0[:2], params)

    scenario_data = {
        "metadata": {
            "name": scenario_name,
            "description": scenario_desc,
            "duration": float(problem.duration),
            "dt": float(problem.dt),
            "n_nodes": int(problem.n_nodes),
            "tau_max": float(problem.tau_max),
            "q_min": float(problem.q_min),
            "q_max": float(problem.q_max),
            "dq_max": float(problem.dq_max),
            "links": {
                "l1": float(params.l1),
                "l2": float(params.l2),
                "m1": float(params.m1),
                "m2": float(params.m2),
                "I1": float(params.I1),
                "I2": float(params.I2),
                "g": float(params.g),
            },
        },
        "target": {
            "q": problem.xf[:2].tolist(),
            "dq": problem.xf[2:].tolist(),
            "ee": target_ee.tolist(),
        },
        "initial": {
            "q": problem.x0[:2].tolist(),
            "dq": problem.x0[2:].tolist(),
            "ee": initial_ee.tolist(),
        },
        "obstacle": None,
        "trajectories": {},
    }

    if obstacle_spec is not None:
        (cx, cy), radius = obstacle_spec
        scenario_data["obstacle"] = {
            "center": [float(cx), float(cy)],
            "radius": float(radius),
        }

    for key, res in results.items():
        t = res.time
        q = res.state[:, :2]
        dq = res.state[:, 2:]
        tau = res.control
        n_pts = len(t)

        p_elbow = np.zeros((n_pts, 2))
        p_ee = np.zeros((n_pts, 2))
        ddq = np.zeros((n_pts, 2))
        tau_inertial = np.zeros((n_pts, 2))
        tau_coriolis = np.zeros((n_pts, 2))
        tau_gravity = np.zeros((n_pts, 2))
        kinetic_energy = np.zeros(n_pts)
        potential_energy = np.zeros(n_pts)
        power_j1 = np.zeros(n_pts)
        power_j2 = np.zeros(n_pts)

        for i in range(n_pts):
            p_elbow[i], p_ee[i] = forward_kinematics(q[i], params)
            ddq[i] = arm.forward_dynamics(q[i], dq[i], tau[i])
            M_i = arm.mass_matrix(q[i])
            C_i = arm.coriolis_vector(q[i], dq[i])
            g_i = arm.gravity_vector(q[i])
            tau_inertial[i] = M_i.dot(ddq[i])
            tau_coriolis[i] = C_i
            tau_gravity[i] = g_i
            kinetic_energy[i] = arm.kinetic_energy(q[i], dq[i])
            potential_energy[i] = arm.potential_energy(q[i])
            power_j1[i] = tau[i, 0] * dq[i, 0]
            power_j2[i] = tau[i, 1] * dq[i, 1]

        reality_data = None
        if res.success:
            rc = perform_reality_check(problem, res, num_eval_points=100)
            ee_drift_sampled = []
            for j in range(len(rc.t_sim)):
                _, p_sim = forward_kinematics(rc.x_sim[j, :2], params)
                _, p_opt = forward_kinematics(rc.x_opt_interp[j, :2], params)
                ee_drift_sampled.append(float(np.linalg.norm(p_sim - p_opt)))
            reality_data = {
                "t_sim": rc.t_sim.tolist(),
                "ee_drift": ee_drift_sampled,
                "max_drift": float(rc.max_cartesian_drift),
                "terminal_drift": float(rc.terminal_cartesian_drift),
            }

        metrics = compute_metrics(problem, res)
        scenario_data["trajectories"][key] = {
            "name": res.method_name,
            "success": bool(res.success),
            "cost": float(res.cost),
            "solve_time": float(res.solve_time),
            "iterations": int(res.iterations),
            "metrics": metrics.to_dict(),
            "time": t.tolist(),
            "q": q.tolist(),
            "dq": dq.tolist(),
            "ddq": ddq.tolist(),
            "tau": tau.tolist(),
            "p_elbow": p_elbow.tolist(),
            "p_ee": p_ee.tolist(),
            "torques_breakdown": {
                "inertial": tau_inertial.tolist(),
                "coriolis": tau_coriolis.tolist(),
                "gravity": tau_gravity.tolist(),
            },
            "energy": {
                "kinetic": kinetic_energy.tolist(),
                "potential": potential_energy.tolist(),
                "total": (kinetic_energy + potential_energy).tolist(),
            },
            "power": {
                "joint1": power_j1.tolist(),
                "joint2": power_j2.tolist(),
                "total": (power_j1 + power_j2).tolist(),
            },
            "reality_check": reality_data,
        }

    return scenario_data


def export_trajectory_json(
    problem: TrajectoryProblem,
    results: Dict[str, TrajectoryResult],
    obstacle_spec: Optional[Tuple[Tuple[float, float], float]] = None,
    scenario_name: str = "Trajectory Optimization",
    scenario_desc: str = "",
    output_path: str = "web_viewer/trajectory_data.json",
):
    data = build_scenario_dict(
        problem,
        results,
        obstacle_spec=obstacle_spec,
        scenario_name=scenario_name,
        scenario_desc=scenario_desc,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
