# Theoretical Foundations of Trajectory Optimization for Planar Manipulators

## 1. System Dynamics of the Planar Two-Link Robot Arm

### 1.1 Generalized Coordinates and State Representation
Consider a two-link planar serial manipulator operating in a vertical gravitational field ($g_0 = 9.81\,\text{m/s}^2$). The arm consists of two rigid links of lengths $l_1, l_2$, masses $m_1, m_2$, with centers of mass located at distances $r_1, r_2$ from their respective proximal joints, and moments of inertia $I_1, I_2$ about their centers of mass.

The generalized coordinates are the relative joint angles:
$$q = \begin{bmatrix} \theta_1 \\ \theta_2 \end{bmatrix} \in \mathbb{R}^2$$

where $\theta_1$ is the angle of link 1 measured counter-clockwise from the positive horizontal $X$-axis, and $\theta_2$ is the relative angle of link 2 with respect to link 1. The state vector $x(t) \in \mathbb{R}^4$ and control input vector $u(t) \in \mathbb{R}^2$ are defined as:
$$x(t) = \begin{bmatrix} q(t) \\ \dot{q}(t) \end{bmatrix} = \begin{bmatrix} \theta_1(t) \\ \theta_2(t) \\ \dot{\theta}_1(t) \\ \dot{\theta}_2(t) \end{bmatrix}, \quad u(t) = \tau(t) = \begin{bmatrix} \tau_1(t) \\ \tau_2(t) \end{bmatrix}$$

### 1.2 Lagrangian Formulation
The equations of motion are derived via Euler-Lagrange equations:
$$\frac{d}{dt} \left( \frac{\partial \mathcal{L}}{\partial \dot{q}} \right) - \frac{\partial \mathcal{L}}{\partial q} = \tau$$

where the Lagrangian $\mathcal{L} = \mathcal{T} - \mathcal{V}$.

#### Kinetic Energy $\mathcal{T}(q, \dot{q})$
The Cartesian positions of the centers of mass are:
$$\begin{aligned}
p_{c1} &= \begin{bmatrix} r_1 \cos\theta_1 \\ r_1 \sin\theta_1 \end{bmatrix} \\
p_{c2} &= \begin{bmatrix} l_1 \cos\theta_1 + r_2 \cos(\theta_1 + \theta_2) \\ l_1 \sin\theta_1 + r_2 \sin(\theta_1 + \theta_2) \end{bmatrix}
\end{aligned}$$

Differentiating with respect to time yields linear velocities $v_{c1} = \dot{p}_{c1}$ and $v_{c2} = \dot{p}_{c2}$. Total kinetic energy includes translational and rotational components:
$$\mathcal{T} = \frac{1}{2} m_1 \|v_{c1}\|^2 + \frac{1}{2} I_1 \dot{\theta}_1^2 + \frac{1}{2} m_2 \|v_{c2}\|^2 + \frac{1}{2} I_2 (\dot{\theta}_1 + \dot{\theta}_2)^2 = \frac{1}{2} \dot{q}^\top M(q) \dot{q}$$

#### Potential Energy $\mathcal{V}(q)$
Taking the datum at $y = 0$:
$$\mathcal{V}(q) = m_1 g_0 r_1 \sin\theta_1 + m_2 g_0 \left( l_1 \sin\theta_1 + r_2 \sin(\theta_1 + \theta_2) \right)$$

### 1.3 Manipulator Equations of Motion
Carrying out the derivatives yields the standard matrix form:
$$M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q) = \tau$$

1. **Inertia Matrix $M(q) = M(q)^\top > 0$**:
   $$M(q) = \begin{bmatrix} M_{11} & M_{12} \\ M_{21} & M_{22} \end{bmatrix}$$
   $$\begin{aligned}
   M_{11}(q) &= m_1 r_1^2 + I_1 + m_2 (l_1^2 + r_2^2 + 2 l_1 r_2 \cos\theta_2) + I_2 \\
   M_{12}(q) &= M_{21}(q) = m_2 (r_2^2 + l_1 r_2 \cos\theta_2) + I_2 \\
   M_{22}(q) &= m_2 r_2^2 + I_2
   \end{aligned}$$

2. **Coriolis and Centrifugal Vector $C(q, \dot{q})\dot{q}$**:
   Letting $h = m_2 l_1 r_2 \sin\theta_2$:
   $$C(q, \dot{q})\dot{q} = \begin{bmatrix} -h (2 \dot{\theta}_1 \dot{\theta}_2 + \dot{\theta}_2^2) \\ h \dot{\theta}_1^2 \end{bmatrix}$$
   The skew-symmetry property $\dot{M}(q) - 2C(q, \dot{q}) = -(\dot{M}(q) - 2C(q, \dot{q}))^\top$ ensures exact power balance:
   $$\frac{d}{dt} \left( \frac{1}{2}\dot{q}^\top M(q)\dot{q} + \mathcal{V}(q) \right) = \dot{q}^\top \tau$$

3. **Gravity Torque Vector $g(q) = \nabla_q \mathcal{V}(q)$**:
   $$g(q) = \begin{bmatrix} (m_1 r_1 + m_2 l_1) g_0 \cos\theta_1 + m_2 r_2 g_0 \cos(\theta_1 + \theta_2) \\ m_2 r_2 g_0 \cos(\theta_1 + \theta_2) \end{bmatrix}$$

4. **Continuous State-Space Representation**:
   $$\dot{x}(t) = f(x(t), u(t)) = \begin{bmatrix} \dot{q}(t) \\ M(q(t))^{-1} \big( u(t) - C(q(t), \dot{q}(t))\dot{q}(t) - g(q(t)) \big) \end{bmatrix}$$

---

## 2. Optimal Control Problem Formulation

The trajectory optimization problem over fixed horizon $t \in [0, T]$ is formulated in the Bolza form:
$$\min_{x(t), u(t)} \quad J = \int_0^T \left( \|u(t)\|^2 + w_{dq} \|\dot{q}(t)\|^2 \right) dt$$
$$\begin{aligned}
\text{subject to} \quad & \dot{x}(t) - f(x(t), u(t)) = 0, && \forall t \in [0, T] \\
& x(0) - x_A = 0, && x(T) - x_B = 0 \\
& -\tau_{\max} \le u(t) \le \tau_{\max}, && \forall t \in [0, T] \\
& q_{\min} \le q(t) \le q_{\max}, && \forall t \in [0, T] \\
& -dq_{\max} \le \dot{q}(t) \le dq_{\max}, && \forall t \in [0, T] \\
& g_{\text{path}}(x(t), u(t)) \ge 0, && \forall t \in [0, T]
\end{aligned}$$

---

## 3. Discretization and Transcription Methods

Direct transcription converts the infinite-dimensional continuous optimal control problem into a finite-dimensional Nonlinear Program (NLP):
$$\min_z \psi(z) \quad \text{s.t.} \quad c_{\text{eq}}(z) = 0, \quad c_{\text{ineq}}(z) \ge 0, \quad z_L \le z \le z_U$$

```
                         TRANSCRIPTION SPECTRUM
                         
   [ Single Shooting ]               [ Trapezoidal Collocation ]          [ Hermite-Simpson Collocation ]
  • Decision: u_0...u_{N-1}          • Decision: x_0,u_0...x_N,u_N        • Decision: x_k, u_k + midpoints
  • ODE simulation in loop           • Piecewise linear dynamics          • Cubic Hermite spline + Simpson
  • O(h^4) RK4 step                  • O(h^2) defect constraints          • O(h^4) defect constraints
  • Dense, sensitive Jacobian        • Sparse block-banded NLP            • Sparse, highly accurate NLP
```

### 3.1 Method 1: Single Direct Shooting (RK4)

* **Decision Vector**: $z = [u_0^\top, u_1^\top, \dots, u_{N-1}^\top]^\top \in \mathbb{R}^{2N}$.
* **Forward Integration**: Given initial condition $x_0 = x_A$, the trajectory is computed sequentially:
  $$x_{k+1} = x_k + \frac{h}{6} (k_1 + 2k_2 + 2k_3 + k_4)$$
  where
  $$\begin{aligned}
  k_1 &= f(x_k, u_k) \\
  k_2 &= f(x_k + \frac{h}{2} k_1, u_k) \\
  k_3 &= f(x_k + \frac{h}{2} k_2, u_k) \\
  k_4 &= f(x_k + h k_3, u_k)
  \end{aligned}$$
* **Terminal Constraint**: Only the endpoint condition is enforced as an NLP constraint:
  $$c_{\text{term}}(z) = x_N(z) - x_B = 0 \in \mathbb{R}^4$$
* **Sensitivity Analysis**:
  Because $x_N$ is formed via repeated function composition $x_N = \phi_N \circ \phi_{N-1} \circ \dots \circ \phi_1(x_0)$, the Jacobian $\frac{\partial x_N}{\partial u_0} = \prod_{k=1}^N \frac{\partial x_{k+1}}{\partial x_k} \frac{\partial x_1}{\partial u_0}$ suffers from multiplicative gradient growth. In unstable or chaotic regimes (e.g. double pendulum), the sensitivity matrix becomes severely ill-conditioned, leading to numerical divergence or getting trapped in poor local extrema.

---

### 3.2 Method 2: Trapezoidal Direct Collocation

Following Cardona-Ortiz & Arechavaleta (2025, Eq. 10-12) and Kelly (2017):
* **Decision Vector**: Both states and controls are optimized simultaneously:
  $$z = [x_0^\top, u_0^\top, x_1^\top, u_1^\top, \dots, x_N^\top, u_N^\top]^\top \in \mathbb{R}^{6(N+1)}$$
* **Defect Constraints**:
  Over each interval $k \in \{0, \dots, N-1\}$ of width $h = T/N$, the continuous dynamics $\dot{x} = f(x, u)$ are integrated using the trapezoidal quadrature rule:
  $$\Delta_k = x_{k+1} - x_k - \frac{h}{2} \big( f(x_k, u_k) + f(x_{k+1}, u_{k+1}) \big) = 0 \in \mathbb{R}^4$$
* **Truncation Error**: The local error is $\mathcal{O}(h^3)$, yielding a global trajectory accuracy of $\mathcal{O}(h^2)$.
* **Cost Function**:
  $$\psi(z) = \sum_{k=0}^{N-1} \frac{h}{2} \left( \|u_k\|^2 + \|u_{k+1}\|^2 \right)$$
* **Sparsity**: Each defect constraint $\Delta_k$ depends only on the adjacent variables $(x_k, u_k, x_{k+1}, u_{k+1})$, generating a block-banded constraint Jacobian that decouples temporal error propagation.

---

### 3.3 Method 3: Hermite–Simpson Direct Collocation (Separated Form)

Following Cardona-Ortiz & Arechavaleta (2025, Eq. 14-16) and Betts (2010):
* **Decision Vector**: Evaluated at $N+1$ mesh nodes and $N$ interior midpoints $\bar{t}_k = t_k + h/2$:
  $$z = [x_0^\top, u_0^\top, \bar{x}_0^\top, \bar{u}_0^\top, x_1^\top, u_1^\top, \dots, \bar{x}_{N-1}^\top, \bar{u}_{N-1}^\top, x_N^\top, u_N^\top]^\top \in \mathbb{R}^{6(2N+1)}$$
* **Defect Constraints**:
  Two conditions are enforced per segment $k$:
  1. **Hermite Midpoint Interpolant**: Forces the midpoint state $\bar{x}_k$ to match the continuous cubic Hermite interpolating polynomial:
     $$\Delta_k^{\text{mid}} = \bar{x}_k - \frac{1}{2}(x_k + x_{k+1}) - \frac{h}{8} \big( f(x_k, u_k) - f(x_{k+1}, u_{k+1}) \big) = 0$$
  2. **Simpson Quadrature Defect**: Enforces dynamic balance across the entire segment using Simpson's rule:
     $$\Delta_k^{\text{dyn}} = x_{k+1} - x_k - \frac{h}{6} \big( f(x_k, u_k) + 4 f(\bar{x}_k, \bar{u}_k) + f(x_{k+1}, u_{k+1}) \big) = 0$$
* **Truncation Error**: The Simpson defect has local error $\mathcal{O}(h^5)$, yielding a global $\mathcal{O}(h^4)$ accuracy (equivalent to 4th-order Runge-Kutta).
* **Cost Function**:
  $$\psi(z) = \sum_{k=0}^{N-1} \frac{h}{6} \left( \|u_k\|^2 + 4 \|\bar{u}_k\|^2 + \|u_{k+1}\|^2 \right)$$

---

## 4. Theoretical Comparison Matrix

| Property | Single Shooting (RK4) | Trapezoidal Collocation | Hermite–Simpson Collocation |
| :--- | :--- | :--- | :--- |
| **Decision Variables** | $2N$ (small) | $6(N+1)$ (moderate) | $12N + 6$ (larger) |
| **NLP Constraint Structure** | Dense terminal (4 equations) | Sparse block-banded ($4N$ eq) | Sparse block-banded ($8N$ eq) |
| **Global Integration Order** | $\mathcal{O}(h^4)$ | $\mathcal{O}(h^2)$ | $\mathcal{O}(h^4)$ |
| **Conditioning of Jacobian** | Exponential decay/growth $\mathcal{O}(e^{\lambda T})$ | Polynomial $\mathcal{O}(h^{-1})$ | Polynomial $\mathcal{O}(h^{-1})$ |
| **Sensitivity to Initial Guess** | Extremely High (requires close controls) | Low (linear state interpolation works) | Low (linear state interpolation works) |
| **Nonlinear Path Constraints** | Difficult (penalty/barrier functions) | Exact algebraic inequalities at nodes | Exact algebraic inequalities at nodes & midpoints |
| **Open-Loop Replay Feasibility** | Exact by construction (in noise-free ODE) | Truncation drift at coarse $N$ | High fidelity even at moderate $N$ |

---

## 5. The Physical Reality Check Methodology

In discrete collocation, satisfaction of the defect constraints $\Delta_k \le \epsilon$ only guarantees that physics is satisfied at the collocation nodes. To rigorously quantify discretization leakage:
1. Extract optimal control sequence $u^*(t_k)$ from the NLP solution.
2. Construct a continuous interpolant $u^*(t) = \text{interp}(t_k, u^*(t_k))$.
3. Forward integrate the continuous dynamics from $x(0) = x_A$ up to $T$ using an independent, high-order adaptive ODE solver (`scipy.integrate.solve_ivp` using the 8th-order Dormand-Prince `DOP853` method with tolerances $\text{rtol}=10^{-8}, \text{atol}=10^{-8}$).
4. Compute the true trajectory drift:
   $$e_{\text{state}}(t) = \| x_{\text{sim}}(t) - x_{\text{opt}}(t) \|, \quad e_{\text{cartesian}}(t) = \| p_{ee}(q_{\text{sim}}(t)) - p_{ee}(q_{\text{opt}}(t)) \|$$

This reality check separates purely mathematical optimization convergence from physical feasibility.

