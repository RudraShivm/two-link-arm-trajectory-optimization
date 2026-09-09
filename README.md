# Two-Link Robot Arm Trajectory Optimization

Comparative benchmark of three numerical trajectory optimization methods applied to a planar two-link robotic manipulator.

## Methods Compared

| # | Method | Transcription | Integration Order |
|---|--------|--------------|-------------------|
| 1 | **Single Direct Shooting** | Controls only; forward RK4 simulation | O(h⁴) per step |
| 2 | **Trapezoidal Direct Collocation** | States + controls; piecewise linear defects | O(h²) global |
| 3 | **Hermite–Simpson Direct Collocation** | States + controls + midpoints; cubic Hermite + Simpson quadrature defects | O(h⁴) global |

## Project Structure

```
arm_opt/
├── dynamics/          # Lagrangian mechanics: M(q), C(q,dq), g(q), kinematics
├── solvers/           # Three trajectory optimization implementations
├── scenarios/         # Benchmark test scenarios
├── analysis/          # Metrics computation and open-loop reality check
└── visualization/     # Matplotlib plots, MP4 animation, web viewer export
tests/                 # Unit tests for dynamics and solver convergence
web_viewer/            # Standalone interactive HTML5 Canvas visualizer
docs/                  # Mathematical derivations and empirical analysis
```

## Quick Start

### Prerequisites

- Python 3.10+
- ffmpeg (for video export)

### Installation

```bash
pip install -r requirements.txt
```

### Run Unit Tests

```bash
PYTHONPATH=. python3 -m unittest discover tests
```

### Run a Benchmark Scenario

```bash
# Single scenario
PYTHONPATH=. python3 run_benchmark.py --scenario rest_to_rest --nodes 30

# All scenarios
PYTHONPATH=. python3 run_benchmark.py --scenario all --nodes 30
```

Available scenarios: `rest_to_rest`, `swing_up`, `obstacle`, `high_speed`

### Generate Showcase Artifacts

Produces comparative plots (PNG), synchronized animation (MP4), and interactive web viewer data (JSON):

```bash
PYTHONPATH=. python3 generate_showcase.py --scenario rest_to_rest --nodes 30
```

### Interactive Web Viewer
### Interactive Research Console (Studio Dashboard)

After generating showcase data, start a local server and open the viewer:
The web dashboard is an editorial-grade research console matching the interface architecture of advanced laboratory benches:

```bash
cd web_viewer
python3 -m http.server 8080
# Open http://localhost:8080 in a browser
# Open http://localhost:8080 in any web browser
```

Features:
- Play/pause with adjustable speed (0.25x, 1x, 2x)
- Time scrubber for frame-by-frame inspection
- Toggle individual methods on/off
- Real-time torque telemetry display
- End-effector trail visualization
- Obstacle rendering (when applicable)
#### Dashboard Features:
- **Pipeline Navigation Tabs**:
  - `Workspace & Kinematics`: 2D robotic arm canvas with antialiased links, joint coordinate frames, link centers of mass, end-effector fading trails, and circular obstacle clearance margins.
  - `State Space & Phase Portraits`: Interactive phase plane orbits $(\theta_1 \text{ vs } \dot{\theta}_1)$ and $(\theta_2 \text{ vs } \dot{\theta}_2)$ tracking the dynamic state manifold.
  - `Dynamic Torques & Power`: Exact physical decomposition into Inertial $M(q)\ddot{q}$, Coriolis/Centrifugal $C(q,\dot{q})\dot{q}$, Gravitational loading $g(q)$, and instantaneous mechanical power $P(t) = \tau^\top \dot{q}$.
  - `Energy Conservation & Hamiltonian`: Real-time tracking of Kinetic energy $\mathcal{T}(t)$, Potential energy $\mathcal{V}(t)$, and Total mechanical energy $\mathcal{E}(t) = \mathcal{T} + \mathcal{V}$.
  - `Physical Reality Check (ODE Drift)`: Continuous 8th-order Runge-Kutta simulation (`DOP853`) drift curves showing how single shooting drifts over time while collocation remains physically bounded.
  - `Comparative Audit Matrix`: High-density quantitative table reporting CPU time, iterations, control effort, peak torque, and drift with automated winner tags.
  - `Mathematical Formulations`: Detailed mathematical defect equations and quadrature rules rendered for each solver.
- **Master Control Rail**:
  - **Scenario Switcher**: Instant switching between *Rest-to-Rest Baseline*, *Cartesian Obstacle Avoidance*, and *High-Speed Coriolis Maneuver*.
  - **Playback Engine**: Play/Pause (Spacebar), step forward/back ($\leftarrow, \rightarrow$), reset ($\circlearrowleft$), time scrubber slider, and speed presets (`0.1x`, `0.25x`, `0.5x`, `1.0x`, `2.0x`).
  - **Real-Time Telemetry HUD**: Joint angles ($\deg$ and $\text{rad}$), angular velocities, linear end-effector speed, and **bilateral centered torque saturation meters** showing instantaneous motor effort relative to $\pm \tau_{\max}$.

## Benchmark Scenarios

### 1. Rest-to-Rest Baseline
Moves from horizontal `[0, 0]` to vertical `[π/2, 0]` in 1.0s. Validates that all solvers converge on a straightforward problem.

### 2. Under-Torqued Swing-Up
Inverts the arm from hanging `[-π/2, 0]` to upright `[π/2, 0]` with torque limited well below the gravitational stall threshold. The arm must pump energy through oscillation. Single Shooting typically fails due to chaotic open-loop sensitivity.

### 3. Cartesian Obstacle Avoidance
Moves between two poses while enforcing end-effector clearance from a circular workspace obstacle. Demonstrates the natural advantage of collocation methods in handling nonlinear path constraints.

### 4. High-Speed Dynamic Maneuver
Executes a large angular stroke in 0.45s where Coriolis and centrifugal terms dominate the dynamics. Highlights the discretization accuracy advantage of Hermite–Simpson over Trapezoidal at coarse grid resolution.

## References

- Cardona-Ortiz, D. and Arechavaleta, G. (2025). *Trajectory optimization for highly articulated robots based on sparsity-free local direct collocation.* Int. J. Appl. Math. Comput. Sci., 35(4), 577–589.
- Kelly, M. (2017). *An Introduction to Trajectory Optimization: How to Do Your Own Direct Collocation.* SIAM Review, 59(4), 849–904.
- Betts, J.T. (2010). *Practical Methods for Optimal Control and Estimation Using Nonlinear Programming.* SIAM.

## Course

CSE 402: Numerical Analysis, Simulation and Modeling Sessional — BUET, Section C-2

