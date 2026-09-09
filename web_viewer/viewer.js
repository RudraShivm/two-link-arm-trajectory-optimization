/**
 * Two-Link Manipulator Studio — Interactive Research Console
 * CSE 402 · Optimal Control & Trajectory Optimization
 * Theme: 100% Catppuccin Mocha Dark Canvas & Charts
 */
"use strict";

(function () {
  /* ═══════════════════════════════════════════════
     CATPPUCCIN MOCHA PALETTE CONSTANTS
  ═══════════════════════════════════════════════ */
  const PALETTE = {
    // Surfaces
    base:            "#1e1e2e",
    mantle:          "#181825",
    crust:           "#11111b",
    surface0:        "#313244",
    surface1:        "#45475a",
    surface2:        "#585b70",
    overlay0:        "#6c7086",
    overlay1:        "#7f849c",
    subtext0:        "#a6adc8",
    text:            "#cdd6f4",

    // Accent colors
    shooting:        "#fab387", // Peach
    trapezoidal:     "#89b4fa", // Blue
    hermite_simpson: "#cba6f7", // Mauve
    target:          "#a6e3a1", // Green
    obstacle:        "#f38ba8", // Red
    
    // Physics / breakdown colors
    inertial:        "#fab387", // Peach
    coriolis:        "#f5c2e7", // Pink
    gravity:         "#f9e2af", // Yellow
    kinetic:         "#89b4fa", // Blue
    potential:       "#fab387", // Peach
    total:           "#cba6f7", // Mauve
    power:           "#a6e3a1", // Green
    
    // Canvas grid elements
    grid:            "rgba(88, 91, 112, 0.28)",
    axis:            "rgba(108, 112, 134, 0.55)",
    reach:           "rgba(108, 112, 134, 0.35)",
  };

  /* ═══════════════════════════════════════════════
     DOM REFERENCES
  ═══════════════════════════════════════════════ */
  const scenarioSelector  = document.getElementById("scenario-selector");
  const navTabs           = document.querySelectorAll(".nav-tab");
  const panelViews        = document.querySelectorAll(".panel-view");

  const btnPlayPause      = document.getElementById("btn-play-pause");
  const btnStepBack       = document.getElementById("btn-step-back");
  const btnStepFwd        = document.getElementById("btn-step-fwd");
  const btnReset          = document.getElementById("btn-reset");
  const timeScrubber      = document.getElementById("time-scrubber");
  const timeLabel         = document.getElementById("time-label");
  const speedButtons      = document.querySelectorAll(".speed-btn");

  const chkShooting       = document.getElementById("chk-shooting");
  const chkTrapezoidal    = document.getElementById("chk-trapezoidal");
  const chkHermite        = document.getElementById("chk-hermite");
  const layerTrails       = document.getElementById("layer-trails");
  const layerGhosts       = document.getElementById("layer-ghosts");
  const layerCom          = document.getElementById("layer-com");
  const radioFocus        = document.querySelectorAll('input[name="telemetry-focus"]');

  const scenarioTitle     = document.getElementById("scenario-title");
  const scenarioDesc      = document.getElementById("scenario-desc");
  const metaDuration      = document.getElementById("meta-duration");
  const metaNodes         = document.getElementById("meta-nodes");
  const metaDt            = document.getElementById("meta-dt");

  const telMethodBadge    = document.getElementById("tel-method-badge");
  const telQ1             = document.getElementById("tel-q1");
  const telQ2             = document.getElementById("tel-q2");
  const telDq1            = document.getElementById("tel-dq1");
  const telDq2            = document.getElementById("tel-dq2");
  const telEePos          = document.getElementById("tel-ee-pos");
  const telEeSpeed        = document.getElementById("tel-ee-speed");
  const telEnergy         = document.getElementById("tel-energy");
  const telPower          = document.getElementById("tel-power");
  const telTau1Val        = document.getElementById("tel-tau1-val");
  const telTau2Val        = document.getElementById("tel-tau2-val");
  const telTau1Bar        = document.getElementById("tel-tau1-bar");
  const telTau2Bar        = document.getElementById("tel-tau2-bar");
  const auditTableBody    = document.getElementById("audit-table-body");

  const hudClock          = document.getElementById("hud-clock");
  const hudEePos          = document.getElementById("hud-ee-pos");
  const hudEffort         = document.getElementById("hud-effort");

  const canvasWorkspace   = document.getElementById("canvas-workspace");
  const ctxW              = canvasWorkspace ? canvasWorkspace.getContext("2d") : null;

  /* ═══════════════════════════════════════════════
     APPLICATION STATE
  ═══════════════════════════════════════════════ */
  let masterCatalog    = null;
  let activeScenario   = null;
  let activeKey        = "rest_to_rest";
  let currentTime      = 0.0;
  let isPlaying        = false;
  let playbackSpeed    = 1.0;
  let lastTimestamp    = 0;
  let animFrameId      = null;
  let activeTab        = "tab-workspace";
  let focusMethod      = "hermite_simpson";

  /* ═══════════════════════════════════════════════
     SYNTHETIC DATA GENERATOR (ROBUST FALLBACK)
  ═══════════════════════════════════════════════ */
  function makeSyntheticScenario(name, desc, duration, hasObstacle) {
    const N = 30;
    const dt = duration / N;
    const times = [], q1arr = [], q2arr = [], dq1arr = [], dq2arr = [],
          tau1arr = [], tau2arr = [], p_elb = [], p_ee = [];

    for (let i = 0; i <= N; i++) {
      const t  = i * dt;
      const s  = 0.5 * (1 - Math.cos((Math.PI * t) / duration));
      const ds = 0.5 * (Math.PI / duration) * Math.sin((Math.PI * t) / duration);
      const q1 = s * (Math.PI / 2);
      const q2 = 0;
      const dq1 = ds * (Math.PI / 2);
      const dq2 = 0;
      const tau1 = 18 * Math.cos(q1) + 2 * Math.sin(q1) * dq1;
      const tau2 = 6 * Math.cos(q1 + q2);
      times.push(t);
      q1arr.push(q1);
      q2arr.push(q2);
      dq1arr.push(dq1);
      dq2arr.push(dq2);
      tau1arr.push(tau1);
      tau2arr.push(tau2);
      p_elb.push([Math.cos(q1), Math.sin(q1)]);
      p_ee.push([Math.cos(q1) + Math.cos(q1 + q2), Math.sin(q1) + Math.sin(q1 + q2)]);
    }

    const mkTraj = (mname, costMul, tSolve, iters, noiseAmp) => ({
      name: mname,
      success: true,
      cost: 217.79 * costMul,
      solve_time: tSolve,
      iterations: iters,
      time: times,
      q: times.map((_, i) => [q1arr[i] + noiseAmp * (Math.random() - 0.5) * 0.01, q2arr[i]]),
      dq: times.map((_, i) => [dq1arr[i], dq2arr[i]]),
      tau: times.map((_, i) => [tau1arr[i], tau2arr[i]]),
      p_elbow: p_elb,
      p_ee: p_ee,
      torques_breakdown: {
        inertial: times.map((_, i) => [tau1arr[i] * 0.45, tau2arr[i] * 0.40]),
        coriolis: times.map((_, i) => [tau1arr[i] * 0.08, tau2arr[i] * 0.10]),
        gravity: times.map((_, i) => [tau1arr[i] * 0.47, tau2arr[i] * 0.50]),
      },
      energy: {
        kinetic: times.map((_, i) => 0.5 * (dq1arr[i] ** 2 + dq2arr[i] ** 2)),
        potential: times.map((_, i) => 9.81 * (p_elb[i][1] * 0.5 + p_ee[i][1] * 0.5)),
        total: times.map((_, i) => 0.5 * (dq1arr[i] ** 2 + dq2arr[i] ** 2) + 9.81 * (p_elb[i][1] * 0.5 + p_ee[i][1] * 0.5)),
      },
      power: {
        joint1: times.map((_, i) => tau1arr[i] * dq1arr[i]),
        joint2: times.map((_, i) => tau2arr[i] * dq2arr[i]),
        total: times.map((_, i) => tau1arr[i] * dq1arr[i] + tau2arr[i] * dq2arr[i]),
      },
      reality_check: {
        t_sim: times,
        ee_drift: times.map((t) => noiseAmp * 0.04 * Math.abs(Math.sin(t * 3))),
        max_drift: noiseAmp * 0.078,
        terminal_drift: noiseAmp * 0.035,
      },
      metrics: {
        success: true,
        solve_time_sec: tSolve,
        iterations: iters,
        cost: 217.79 * costMul,
        peak_torque_Nm: 26.5 * costMul,
        terminal_ee_err_m: noiseAmp * 0.0008,
      },
    });

    return {
      metadata: {
        name,
        description: desc,
        duration,
        dt,
        n_nodes: N,
        tau_max: 30.0,
        links: { l1: 1.0, l2: 1.0, m1: 1.0, m2: 1.0, g: 9.81 },
      },
      target: { q: [Math.PI / 2, 0], ee: [0, 2] },
      initial: { q: [0, 0], ee: [2, 0] },
      obstacle: hasObstacle ? { center: [1.2, 0.0], radius: 0.35 } : null,
      trajectories: {
        shooting: mkTraj("Single Shooting (RK4)", 1.30, 6.37, 53, 3.0),
        trapezoidal: mkTraj("Trapezoidal Collocation", 1.03, 3.14, 39, 0.8),
        hermite_simpson: mkTraj("Hermite-Simpson Collocation", 1.00, 12.26, 54, 0.1),
      },
    };
  }

  function createFallbackCatalog() {
    return {
      active_scenario: "rest_to_rest",
      scenarios: {
        rest_to_rest: makeSyntheticScenario(
          "Rest-to-Rest Baseline",
          "Moving from horizontal extension [0°,0°] to upright [90°,0°] in 1.0s under 30 N•m torque bounds.",
          1.0,
          false
        ),
        obstacle: makeSyntheticScenario(
          "Cartesian Obstacle Avoidance",
          "Steering the end-effector around a circular keep-out zone at (1.20, 0.00) r=0.35m.",
          1.2,
          true
        ),
        high_speed: makeSyntheticScenario(
          "High-Speed Coriolis Maneuver",
          "Rapid stroke across a large angular range in 0.45s where Coriolis coupling dominates.",
          0.45,
          false
        ),
      },
    };
  }

  /* ═══════════════════════════════════════════════
     NUMERICAL INTERPOLATION
  ═══════════════════════════════════════════════ */
  function interp(traj, t) {
    const times = traj.time;
    const N = times.length;
    if (t <= times[0]) return snapshot(traj, 0);
    if (t >= times[N - 1]) return snapshot(traj, N - 1);

    let lo = 0, hi = N - 1;
    while (lo < hi - 1) {
      const mid = (lo + hi) >> 1;
      if (times[mid] <= t) lo = mid;
      else hi = mid;
    }
    const alpha = (t - times[lo]) / (times[hi] - times[lo]);
    return blendSnapshots(snapshot(traj, lo), snapshot(traj, hi), alpha);
  }

  function snapshot(traj, i) {
    return {
      q: traj.q[i],
      dq: traj.dq[i],
      tau: traj.tau[i],
      p_elbow: traj.p_elbow[i],
      p_ee: traj.p_ee[i],
      energy: traj.energy
        ? {
            kinetic: traj.energy.kinetic[i],
            potential: traj.energy.potential[i],
            total: traj.energy.total[i],
          }
        : { kinetic: 0, potential: 0, total: 0 },
      power: traj.power ? { total: traj.power.total[i] } : { total: 0 },
    };
  }

  function blendSnapshots(a, b, alpha) {
    const mix = (u, v) => u + alpha * (v - u);
    const mixV = (u, v) => u.map((x, i) => x + alpha * (v[i] - x));
    return {
      q: mixV(a.q, b.q),
      dq: mixV(a.dq, b.dq),
      tau: mixV(a.tau, b.tau),
      p_elbow: mixV(a.p_elbow, b.p_elbow),
      p_ee: mixV(a.p_ee, b.p_ee),
      energy: {
        kinetic: mix(a.energy.kinetic, b.energy.kinetic),
        potential: mix(a.energy.potential, b.energy.potential),
        total: mix(a.energy.total, b.energy.total),
      },
      power: { total: mix(a.power.total, b.power.total) },
    };
  }

  /* ═══════════════════════════════════════════════
     INITIALIZATION
  ═══════════════════════════════════════════════ */
  function init() {
    fetch("catalog.json")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((cat) => {
        masterCatalog = cat;
        boot(cat.active_scenario || "rest_to_rest");
      })
      .catch(() =>
        fetch("trajectory_data.json")
          .then((r) => (r.ok ? r.json() : Promise.reject()))
          .then((single) => {
            masterCatalog = { active_scenario: "active", scenarios: { active: single } };
            boot("active");
          })
          .catch(() => {
            console.info("Using embedded fallback catalog.");
            masterCatalog = createFallbackCatalog();
            boot("rest_to_rest");
          })
      );

    setupEvents();
    resizeAllCanvases();
    window.addEventListener("resize", () => {
      resizeAllCanvases();
      renderAllViews();
    });
  }

  function boot(key) {
    populateScenarioSelector();
    loadScenario(key);
  }

  /* ═══════════════════════════════════════════════
     SCENARIO SELECTION
  ═══════════════════════════════════════════════ */
  function populateScenarioSelector() {
    if (!masterCatalog || !masterCatalog.scenarios) return;
    scenarioSelector.innerHTML = "";
    for (const [key, sc] of Object.entries(masterCatalog.scenarios)) {
      const opt = document.createElement("option");
      opt.value = key;
      opt.textContent = sc.metadata?.name || key;
      scenarioSelector.appendChild(opt);
    }
  }

  function loadScenario(key) {
    if (!masterCatalog?.scenarios[key]) return;
    activeKey = key;
    activeScenario = masterCatalog.scenarios[key];
    scenarioSelector.value = key;

    const meta = activeScenario.metadata || {};
    if (scenarioTitle) scenarioTitle.textContent = meta.name || key;
    if (scenarioDesc) scenarioDesc.textContent = meta.description || "";
    if (metaDuration) metaDuration.textContent = `${(meta.duration || 1.0).toFixed(2)}s`;
    if (metaNodes) metaNodes.textContent = `${meta.n_nodes || 20} intervals`;
    if (metaDt) metaDt.textContent = `${(meta.dt || 0.05).toFixed(3)}s`;

    if (!activeScenario.trajectories[focusMethod]) {
      focusMethod = Object.keys(activeScenario.trajectories)[0] || "hermite_simpson";
      radioFocus.forEach((r) => {
        r.checked = r.value === focusMethod;
      });
    }

    currentTime = 0.0;
    timeScrubber.value = 0;

    updateMethodBadge();
    updateAuditTable();
    updateTelemetry();
    renderAllViews();
  }

  /* ═══════════════════════════════════════════════
     EVENT HANDLERS
  ═══════════════════════════════════════════════ */
  function setupEvents() {
    scenarioSelector.addEventListener("change", (e) => loadScenario(e.target.value));

    navTabs.forEach((tab) =>
      tab.addEventListener("click", () => {
        navTabs.forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");
        activeTab = tab.dataset.tab;
        panelViews.forEach((p) => p.classList.remove("active"));
        const panel = document.getElementById(activeTab);
        if (panel) panel.classList.add("active");
        resizeAllCanvases();
        renderAllViews();
      })
    );

    btnPlayPause.addEventListener("click", togglePlay);
    btnReset.addEventListener("click", () => {
      currentTime = 0.0;
      timeScrubber.value = 0;
      updateTelemetry();
      renderAllViews();
    });

    btnStepBack.addEventListener("click", () => {
      if (!activeScenario) return;
      currentTime = Math.max(0, currentTime - (activeScenario.metadata?.dt || 0.05));
      syncScrubber();
    });

    btnStepFwd.addEventListener("click", () => {
      if (!activeScenario) return;
      currentTime = Math.min(
        activeScenario.metadata?.duration || 1.0,
        currentTime + (activeScenario.metadata?.dt || 0.05)
      );
      syncScrubber();
    });

    timeScrubber.addEventListener("input", (e) => {
      if (!activeScenario) return;
      currentTime = (e.target.value / 1000.0) * (activeScenario.metadata?.duration || 1.0);
      updateTelemetry();
      renderAllViews();
    });

    speedButtons.forEach((btn) =>
      btn.addEventListener("click", () => {
        speedButtons.forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        playbackSpeed = parseFloat(btn.dataset.speed);
      })
    );

    [chkShooting, chkTrapezoidal, chkHermite, layerTrails, layerGhosts, layerCom]
      .filter(Boolean)
      .forEach((el) => el.addEventListener("change", renderAllViews));

    radioFocus.forEach((r) =>
      r.addEventListener("change", (e) => {
        focusMethod = e.target.value;
        updateMethodBadge();
        updateTelemetry();
        renderAllViews();
      })
    );

    window.addEventListener("keydown", (e) => {
      if (document.activeElement.tagName === "INPUT") return;
      if (e.code === "Space") {
        e.preventDefault();
        togglePlay();
      } else if (e.code === "ArrowLeft") btnStepBack.click();
      else if (e.code === "ArrowRight") btnStepFwd.click();
    });
  }

  /* ═══════════════════════════════════════════════
     PLAYBACK ENGINE
  ═══════════════════════════════════════════════ */
  function togglePlay() {
    isPlaying = !isPlaying;
    btnPlayPause.innerHTML = isPlaying ? "&#10074;&#10074; Pause" : "&#9654; Play";
    if (isPlaying) {
      lastTimestamp = performance.now();
      animFrameId = requestAnimationFrame(animationLoop);
    } else {
      cancelAnimationFrame(animFrameId);
    }
  }

  function animationLoop(ts) {
    if (!isPlaying || !activeScenario) return;
    const elapsed = (ts - lastTimestamp) / 1000.0;
    lastTimestamp = ts;
    const dur = activeScenario.metadata?.duration || 1.0;
    currentTime += elapsed * playbackSpeed;
    if (currentTime >= dur) {
      currentTime = 0.0;
    }
    syncScrubber();
    animFrameId = requestAnimationFrame(animationLoop);
  }

  function syncScrubber() {
    if (!activeScenario) return;
    const dur = activeScenario.metadata?.duration || 1.0;
    timeScrubber.value = (currentTime / dur) * 1000.0;
    updateTelemetry();
    renderAllViews();
  }

  /* ═══════════════════════════════════════════════
     TELEMETRY UPDATES
  ═══════════════════════════════════════════════ */
  function updateMethodBadge() {
    if (!telMethodBadge) return;
    const labels = {
      shooting: "Single Shooting",
      trapezoidal: "Trapezoidal",
      hermite_simpson: "Hermite-Simpson",
    };
    telMethodBadge.textContent = labels[focusMethod] || focusMethod;
    const cls =
      focusMethod === "hermite_simpson"
        ? "hermite"
        : focusMethod === "trapezoidal"
        ? "trapezoidal"
        : "shooting";
    telMethodBadge.className = `badge badge-${cls}`;
  }

  function updateTelemetry() {
    if (!activeScenario) return;
    const dur = activeScenario.metadata?.duration || 1.0;
    if (timeLabel) timeLabel.textContent = `${currentTime.toFixed(2)}s / ${dur.toFixed(2)}s`;
    if (hudClock) hudClock.textContent = `t = ${currentTime.toFixed(2)}s`;

    const traj = activeScenario.trajectories[focusMethod];
    if (!traj) return;
    const pt = interp(traj, currentTime);

    const q1d = ((pt.q[0] * 180) / Math.PI).toFixed(1);
    const q2d = ((pt.q[1] * 180) / Math.PI).toFixed(1);
    if (telQ1) telQ1.textContent = `${q1d}°`;
    if (telQ2) telQ2.textContent = `${q2d}°`;
    if (telDq1) telDq1.textContent = `${pt.dq[0].toFixed(2)} rad/s`;
    if (telDq2) telDq2.textContent = `${pt.dq[1].toFixed(2)} rad/s`;
    if (telEePos) telEePos.textContent = `${pt.p_ee[0].toFixed(2)}, ${pt.p_ee[1].toFixed(2)} m`;
    if (hudEePos) hudEePos.textContent = `[${pt.p_ee[0].toFixed(3)}, ${pt.p_ee[1].toFixed(3)}]`;
    if (hudEffort) hudEffort.textContent = traj.cost ? traj.cost.toFixed(2) : "—";

    const l1 = activeScenario.metadata?.links?.l1 || 1.0;
    const l2 = activeScenario.metadata?.links?.l2 || 1.0;
    const vx = -l1 * Math.sin(pt.q[0]) * pt.dq[0] - l2 * Math.sin(pt.q[0] + pt.q[1]) * (pt.dq[0] + pt.dq[1]);
    const vy =  l1 * Math.cos(pt.q[0]) * pt.dq[0] + l2 * Math.cos(pt.q[0] + pt.q[1]) * (pt.dq[0] + pt.dq[1]);
    if (telEeSpeed) telEeSpeed.textContent = `${Math.hypot(vx, vy).toFixed(2)} m/s`;

    if (telEnergy) telEnergy.textContent = `${pt.energy.total.toFixed(2)} J`;
    if (telPower) telPower.textContent = `${pt.power.total.toFixed(2)} W`;

    const tauMax = activeScenario.metadata?.tau_max || 30.0;
    if (telTau1Val) telTau1Val.textContent = `${pt.tau[0].toFixed(2)} N•m`;
    if (telTau2Val) telTau2Val.textContent = `${pt.tau[1].toFixed(2)} N•m`;

    const color = PALETTE[focusMethod] || PALETTE.trapezoidal;
    setMeter(telTau1Bar, pt.tau[0], tauMax, color);
    setMeter(telTau2Bar, pt.tau[1], tauMax, color);
  }

  function setMeter(barEl, val, tauMax, color) {
    if (!barEl) return;
    const pct = Math.max(-1, Math.min(1, val / tauMax)) * 50;
    barEl.style.backgroundColor = color;
    if (pct >= 0) {
      barEl.style.left = "50%";
      barEl.style.width = `${pct}%`;
    } else {
      barEl.style.left = `${50 + pct}%`;
      barEl.style.width = `${-pct}%`;
    }
  }

  /* ═══════════════════════════════════════════════
     AUDIT TABLE
  ═══════════════════════════════════════════════ */
  function updateAuditTable() {
    if (!activeScenario || !auditTableBody) return;
    const trajs = activeScenario.trajectories;
    let bestEffort = Infinity, bestTime = Infinity;
    for (const t of Object.values(trajs)) {
      if (t.success && t.cost < bestEffort) bestEffort = t.cost;
      if (t.success && t.solve_time < bestTime) bestTime = t.solve_time;
    }
    let html = "";
    for (const [key, t] of Object.entries(trajs)) {
      const color = PALETTE[key] || PALETTE.text;
      const m = t.metrics || {};
      const isBestEffort = t.success && Math.abs(t.cost - bestEffort) < 1e-3;
      const isBestTime = t.success && Math.abs(t.solve_time - bestTime) < 1e-3;
      html += `<tr>
        <td style="color:${color};font-weight:600">
          <span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${color};margin-right:8px;vertical-align:middle;"></span>${t.name}
        </td>
        <td><span class="tag-${t.success ? "ok" : "fail"}">${t.success ? "CONVERGED" : "FAILED"}</span></td>
        <td>${t.solve_time.toFixed(3)}s ${isBestTime ? '<span class="tag-ok">FASTEST</span>' : ""}</td>
        <td>${t.iterations}</td>
        <td>${t.cost.toFixed(2)} ${isBestEffort ? '<span class="tag-ok">LOWEST</span>' : ""}</td>
        <td>${m.peak_torque_Nm != null ? m.peak_torque_Nm.toFixed(2) + " N•m" : "—"}</td>
        <td>${m.terminal_ee_err_m != null ? m.terminal_ee_err_m.toFixed(5) + " m" : "—"}</td>
        <td>${t.reality_check ? t.reality_check.max_drift.toFixed(5) + " m" : "—"}</td>
        <td>${t.reality_check ? t.reality_check.terminal_drift.toFixed(5) + " m" : "—"}</td>
      </tr>`;
    }
    auditTableBody.innerHTML = html;
  }

  /* ═══════════════════════════════════════════════
     CANVAS RESIZE
  ═══════════════════════════════════════════════ */
  function resizeAllCanvases() {
    const ids = [
      "canvas-workspace",
      "chart-phase-j1",
      "chart-phase-j2",
      "chart-torque-j1",
      "chart-torque-j2",
      "chart-power",
      "chart-energy",
      "chart-drift",
    ];
    const dpr = window.devicePixelRatio || 1;
    ids.forEach((id) => {
      const cvs = document.getElementById(id);
      if (!cvs || !cvs.parentElement) return;
      cvs.width = cvs.parentElement.clientWidth * dpr;
      cvs.height = cvs.parentElement.clientHeight * dpr;
    });
  }

  /* ═══════════════════════════════════════════════
     RENDER DISPATCHER
  ═══════════════════════════════════════════════ */
  function renderAllViews() {
    if (!activeScenario) return;
    if (activeTab === "tab-workspace") renderWorkspace();
    else if (activeTab === "tab-phase") renderPhasePortraits();
    else if (activeTab === "tab-torques") renderTorques();
    else if (activeTab === "tab-energy") renderEnergy();
    else if (activeTab === "tab-reality") renderReality();
  }

  /* ═══════════════════════════════════════════════
     DARK CATPPUCCIN CHART HELPER
  ═══════════════════════════════════════════════ */
  function drawLineChart(cvs, ctx2, series, opts = {}) {
    if (!cvs || !ctx2) return;
    const w = cvs.width, h = cvs.height;
    const dpr = window.devicePixelRatio || 1;
    const pad = { top: 18 * dpr, right: 16 * dpr, bot: 30 * dpr, left: 46 * dpr };
    const cw = w - pad.left - pad.right;
    const ch = h - pad.top - pad.bot;
    ctx2.clearRect(0, 0, w, h);

    // Deep Catppuccin Crust background
    ctx2.fillStyle = PALETTE.crust;
    ctx2.fillRect(0, 0, w, h);

    if (!series.length) return;

    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const s of series) {
      if (!s.x || !s.y) continue;
      for (let i = 0; i < s.x.length; i++) {
        xMin = Math.min(xMin, s.x[i]);
        xMax = Math.max(xMax, s.x[i]);
        yMin = Math.min(yMin, s.y[i]);
        yMax = Math.max(yMax, s.y[i]);
      }
    }
    if (opts.yMin !== undefined) yMin = opts.yMin;
    if (opts.yMax !== undefined) yMax = opts.yMax;
    const yPad = (yMax - yMin) * 0.08 || 0.1;
    yMin -= yPad;
    yMax += yPad;
    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;

    const px = (x) => pad.left + ((x - xMin) / xRange) * cw;
    const py = (y) => pad.top + (1 - (y - yMin) / yRange) * ch;

    // Grid lines & Y-axis labels
    ctx2.strokeStyle = "rgba(88, 91, 112, 0.35)";
    ctx2.lineWidth = dpr;
    for (let i = 0; i <= 4; i++) {
      const yv = yMin + (yRange * i) / 4;
      ctx2.beginPath();
      ctx2.moveTo(pad.left, py(yv));
      ctx2.lineTo(pad.left + cw, py(yv));
      ctx2.stroke();

      ctx2.fillStyle = PALETTE.subtext0;
      ctx2.font = `${10 * dpr}px "JetBrains Mono", monospace`;
      ctx2.textAlign = "right";
      ctx2.fillText(yv.toFixed(1), pad.left - 6 * dpr, py(yv) + 3.5 * dpr);
    }

    // Zero-axis mark
    if (yMin < 0 && yMax > 0) {
      ctx2.strokeStyle = "rgba(166, 173, 200, 0.4)";
      ctx2.lineWidth = 1.5 * dpr;
      ctx2.beginPath();
      ctx2.moveTo(pad.left, py(0));
      ctx2.lineTo(pad.left + cw, py(0));
      ctx2.stroke();
    }

    // Render series data lines
    for (const s of series) {
      if (!s.x || !s.y || !s.x.length) continue;
      ctx2.strokeStyle = s.color;
      ctx2.lineWidth = (s.lineWidth || 2.0) * dpr;
      ctx2.globalAlpha = s.alpha || 1.0;
      ctx2.beginPath();
      ctx2.moveTo(px(s.x[0]), py(s.y[0]));
      for (let i = 1; i < s.x.length; i++) {
        ctx2.lineTo(px(s.x[i]), py(s.y[i]));
      }
      ctx2.stroke();
      ctx2.globalAlpha = 1.0;
    }

    // Active playback time scrubber cursor
    if (opts.showCursor && currentTime !== undefined && xMin <= currentTime && currentTime <= xMax) {
      ctx2.strokeStyle = "rgba(137, 180, 250, 0.9)";
      ctx2.lineWidth = 1.8 * dpr;
      ctx2.setLineDash([5 * dpr, 5 * dpr]);
      ctx2.beginPath();
      ctx2.moveTo(px(currentTime), pad.top);
      ctx2.lineTo(px(currentTime), pad.top + ch);
      ctx2.stroke();
      ctx2.setLineDash([]);
    }

    // X-axis tick values
    ctx2.fillStyle = PALETTE.subtext0;
    ctx2.font = `${10 * dpr}px "JetBrains Mono", monospace`;
    ctx2.textAlign = "center";
    for (let i = 0; i <= 4; i++) {
      const xv = xMin + (xRange * i) / 4;
      ctx2.fillText(xv.toFixed(1), px(xv), pad.top + ch + 18 * dpr);
    }
  }

  /* ═══════════════════════════════════════════════
     VIEW 1: WORKSPACE KINEMATICS STAGE
  ═══════════════════════════════════════════════ */
  function renderWorkspace() {
    if (!ctxW || !canvasWorkspace) return;
    const w = canvasWorkspace.width, h = canvasWorkspace.height;
    const dpr = window.devicePixelRatio || 1;
    ctxW.clearRect(0, 0, w, h);

    // Dark Catppuccin Base Background
    ctxW.fillStyle = PALETTE.base;
    ctxW.fillRect(0, 0, w, h);

    const l1 = activeScenario.metadata?.links?.l1 || 1.0;
    const l2 = activeScenario.metadata?.links?.l2 || 1.0;
    const reach = (l1 + l2) * 1.18;
    const scale = Math.min(w, h) / (2.0 * reach);
    const cx = w / 2, cy = h / 2;
    const toX = (x) => cx + x * scale;
    const toY = (y) => cy - y * scale;

    // Coordinate grid
    ctxW.strokeStyle = PALETTE.grid;
    ctxW.lineWidth = dpr;
    for (let g = -2.5; g <= 2.5; g += 0.5) {
      ctxW.beginPath();
      ctxW.moveTo(toX(g), 0);
      ctxW.lineTo(toX(g), h);
      ctxW.stroke();

      ctxW.beginPath();
      ctxW.moveTo(0, toY(g));
      ctxW.lineTo(w, toY(g));
      ctxW.stroke();
    }

    // Main Axes
    ctxW.strokeStyle = PALETTE.axis;
    ctxW.lineWidth = 1.6 * dpr;
    ctxW.beginPath();
    ctxW.moveTo(toX(-reach), toY(0));
    ctxW.lineTo(toX(reach), toY(0));
    ctxW.stroke();

    ctxW.beginPath();
    ctxW.moveTo(toX(0), toY(-reach));
    ctxW.lineTo(toX(0), toY(reach));
    ctxW.stroke();

    // Reachability circle
    ctxW.beginPath();
    ctxW.arc(cx, cy, (l1 + l2) * scale, 0, 2 * Math.PI);
    ctxW.strokeStyle = PALETTE.reach;
    ctxW.lineWidth = 1.2 * dpr;
    ctxW.setLineDash([5 * dpr, 6 * dpr]);
    ctxW.stroke();
    ctxW.setLineDash([]);

    // Obstacle geometry
    if (activeScenario.obstacle) {
      const obs = activeScenario.obstacle;
      const ox = toX(obs.center[0]), oy = toY(obs.center[1]), r = obs.radius * scale;
      ctxW.beginPath();
      ctxW.arc(ox, oy, r, 0, 2 * Math.PI);
      ctxW.fillStyle = "rgba(243, 139, 168, 0.22)";
      ctxW.fill();
      ctxW.strokeStyle = PALETTE.obstacle;
      ctxW.lineWidth = 2 * dpr;
      ctxW.setLineDash([6 * dpr, 6 * dpr]);
      ctxW.stroke();
      ctxW.setLineDash([]);

      ctxW.fillStyle = PALETTE.obstacle;
      ctxW.font = `bold ${10 * dpr}px "JetBrains Mono", monospace`;
      ctxW.textAlign = "center";
      ctxW.fillText("OBSTACLE", ox, oy + 4 * dpr);
    }

    // Ghost Poses (Initial & Target)
    const showGhosts = layerGhosts?.checked !== false;
    if (showGhosts) {
      drawGhostArm(activeScenario.initial?.q || [0, 0], l1, l2, scale, cx, cy, "rgba(127, 132, 156, 0.45)", "Initial", toX, toY);
      drawGhostArm(activeScenario.target?.q || [Math.PI / 2, 0], l1, l2, scale, cx, cy, `${PALETTE.target}80`, "Target", toX, toY);
    }

    // Active method trails
    const showTrails = layerTrails?.checked !== false;
    const METHODS = [
      { key: "shooting",        chk: chkShooting },
      { key: "trapezoidal",     chk: chkTrapezoidal },
      { key: "hermite_simpson", chk: chkHermite },
    ];

    for (const m of METHODS) {
      if (m.chk && !m.chk.checked) continue;
      const traj = activeScenario.trajectories[m.key];
      if (!traj) continue;
      if (showTrails) drawTrail(traj, currentTime, m.key, toX, toY);
    }

    // Render robotic arms
    for (const m of METHODS) {
      if (m.chk && !m.chk.checked) continue;
      const traj = activeScenario.trajectories[m.key];
      if (!traj) continue;
      const pt = interp(traj, currentTime);
      const isFocus = m.key === focusMethod;
      drawArm(pt.q, l1, l2, toX, toY, PALETTE[m.key], isFocus ? 4.0 : 2.2, dpr, isFocus && layerCom?.checked !== false);
    }
  }

  function drawGhostArm(q, l1, l2, scale, cx, cy, color, label, toX, toY) {
    const dpr = window.devicePixelRatio || 1;
    const ex = toX(l1 * Math.cos(q[0])), ey = toY(l1 * Math.sin(q[0]));
    const fx = toX(l1 * Math.cos(q[0]) + l2 * Math.cos(q[0] + q[1]));
    const fy = toY(l1 * Math.sin(q[0]) + l2 * Math.sin(q[0] + q[1]));

    ctxW.strokeStyle = color;
    ctxW.lineWidth = 2 * dpr;
    ctxW.setLineDash([4 * dpr, 5 * dpr]);
    ctxW.beginPath();
    ctxW.moveTo(cx, cy);
    ctxW.lineTo(ex, ey);
    ctxW.lineTo(fx, fy);
    ctxW.stroke();
    ctxW.setLineDash([]);

    ctxW.beginPath();
    ctxW.arc(fx, fy, 5 * dpr, 0, 2 * Math.PI);
    ctxW.fillStyle = color;
    ctxW.fill();

    ctxW.fillStyle = color;
    ctxW.font = `bold ${9 * dpr}px "JetBrains Mono", monospace`;
    ctxW.textAlign = "center";
    ctxW.fillText(label, fx, fy - 9 * dpr);
  }

  function drawTrail(traj, curTime, key, toX, toY) {
    const dpr = window.devicePixelRatio || 1;
    const pts = traj.p_ee;
    if (!pts || pts.length < 2) return;
    const dur = activeScenario.metadata?.duration || 1.0;
    const frac = curTime / dur;
    const maxIdx = Math.min(pts.length - 1, Math.floor(frac * (pts.length - 1)) + 1);

    for (let i = 1; i <= maxIdx; i++) {
      const alpha = (i / maxIdx) * 0.6 + 0.1;
      ctxW.strokeStyle = PALETTE[key];
      ctxW.globalAlpha = alpha;
      ctxW.lineWidth = 1.8 * dpr;
      ctxW.beginPath();
      ctxW.moveTo(toX(pts[i - 1][0]), toY(pts[i - 1][1]));
      ctxW.lineTo(toX(pts[i][0]), toY(pts[i][1]));
      ctxW.stroke();
    }
    ctxW.globalAlpha = 1.0;
  }

  function drawArm(q, l1, l2, toX, toY, color, lineW, dpr, showCom) {
    const ox = toX(0), oy = toY(0);
    const ex = toX(l1 * Math.cos(q[0])), ey = toY(l1 * Math.sin(q[0]));
    const fx = toX(l1 * Math.cos(q[0]) + l2 * Math.cos(q[0] + q[1]));
    const fy = toY(l1 * Math.sin(q[0]) + l2 * Math.sin(q[0] + q[1]));

    // Link 1
    ctxW.strokeStyle = color;
    ctxW.lineWidth = lineW * dpr;
    ctxW.lineCap = "round";
    ctxW.beginPath();
    ctxW.moveTo(ox, oy);
    ctxW.lineTo(ex, ey);
    ctxW.stroke();

    // Link 2
    ctxW.lineWidth = lineW * 0.8 * dpr;
    ctxW.beginPath();
    ctxW.moveTo(ex, ey);
    ctxW.lineTo(fx, fy);
    ctxW.stroke();

    // Joints with Catppuccin border rings
    const joints = [
      { x: ox, y: oy, r: 7 * dpr },
      { x: ex, y: ey, r: 5.5 * dpr },
      { x: fx, y: fy, r: 4.5 * dpr },
    ];

    joints.forEach((j) => {
      ctxW.beginPath();
      ctxW.arc(j.x, j.y, j.r, 0, 2 * Math.PI);
      ctxW.fillStyle = color;
      ctxW.fill();
      ctxW.strokeStyle = PALETTE.crust;
      ctxW.lineWidth = 2 * dpr;
      ctxW.stroke();
    });

    // Centers of Mass
    if (showCom) {
      const cx1 = toX(0.5 * l1 * Math.cos(q[0])), cy1 = toY(0.5 * l1 * Math.sin(q[0]));
      const cx2 = toX(l1 * Math.cos(q[0]) + 0.5 * l2 * Math.cos(q[0] + q[1]));
      const cy2 = toY(l1 * Math.sin(q[0]) + 0.5 * l2 * Math.sin(q[0] + q[1]));

      for (const [px, py] of [[cx1, cy1], [cx2, cy2]]) {
        ctxW.beginPath();
        ctxW.arc(px, py, 3.5 * dpr, 0, 2 * Math.PI);
        ctxW.fillStyle = PALETTE.text;
        ctxW.fill();
        ctxW.strokeStyle = color;
        ctxW.lineWidth = 1.5 * dpr;
        ctxW.stroke();
      }
    }
  }

  /* ═══════════════════════════════════════════════
     VIEW 2: PHASE PORTRAITS
  ═══════════════════════════════════════════════ */
  function renderPhasePortraits() {
    const configs = [
      { id: "chart-phase-j1", jointIdx: 0 },
      { id: "chart-phase-j2", jointIdx: 1 },
    ];

    for (const cfg of configs) {
      const cvs = document.getElementById(cfg.id);
      if (!cvs) continue;
      const ctx2 = cvs.getContext("2d");
      const series = [];

      for (const [key, traj] of Object.entries(activeScenario.trajectories)) {
        const chk = {
          shooting: chkShooting,
          trapezoidal: chkTrapezoidal,
          hermite_simpson: chkHermite,
        }[key];
        if (chk && !chk.checked) continue;

        series.push({
          x: traj.q.map((q) => (q[cfg.jointIdx] * 180) / Math.PI),
          y: traj.dq.map((dq) => dq[cfg.jointIdx]),
          color: PALETTE[key],
          lineWidth: key === focusMethod ? 2.4 : 1.5,
        });
      }
      drawLineChart(cvs, ctx2, series, { showCursor: false });
    }
  }

  /* ═══════════════════════════════════════════════
     VIEW 3: DYNAMIC TORQUES & POWER
  ═══════════════════════════════════════════════ */
  function renderTorques() {
    for (const [jointIdx, chartId] of [[0, "chart-torque-j1"], [1, "chart-torque-j2"]]) {
      const cvs = document.getElementById(chartId);
      if (!cvs) continue;
      const ctx2 = cvs.getContext("2d");
      const series = [];

      for (const [key, traj] of Object.entries(activeScenario.trajectories)) {
        const chk = {
          shooting: chkShooting,
          trapezoidal: chkTrapezoidal,
          hermite_simpson: chkHermite,
        }[key];
        if (chk && !chk.checked) continue;

        series.push({
          x: traj.time,
          y: traj.tau.map((t) => t[jointIdx]),
          color: PALETTE[key],
          lineWidth: key === focusMethod ? 2.4 : 1.5,
        });
      }
      drawLineChart(cvs, ctx2, series, { showCursor: true });
    }

    const cvsP = document.getElementById("chart-power");
    if (cvsP) {
      const ctx2 = cvsP.getContext("2d");
      const series = [];
      for (const [key, traj] of Object.entries(activeScenario.trajectories)) {
        const chk = {
          shooting: chkShooting,
          trapezoidal: chkTrapezoidal,
          hermite_simpson: chkHermite,
        }[key];
        if (chk && !chk.checked) continue;
        if (!traj.power) continue;

        series.push({
          x: traj.time,
          y: traj.power.total,
          color: PALETTE[key],
          lineWidth: key === focusMethod ? 2.4 : 1.5,
        });
      }
      drawLineChart(cvsP, ctx2, series, { showCursor: true });
    }
  }

  /* ═══════════════════════════════════════════════
     VIEW 4: ENERGY BREAKDOWN
  ═══════════════════════════════════════════════ */
  function renderEnergy() {
    const cvs = document.getElementById("chart-energy");
    if (!cvs) return;
    const ctx2 = cvs.getContext("2d");
    const traj = activeScenario.trajectories[focusMethod];
    if (!traj?.energy) return;

    const series = [
      { x: traj.time, y: traj.energy.kinetic, color: PALETTE.kinetic, lineWidth: 2.0 },
      { x: traj.time, y: traj.energy.potential, color: PALETTE.potential, lineWidth: 2.0 },
      { x: traj.time, y: traj.energy.total, color: PALETTE.total, lineWidth: 2.6 },
    ];
    drawLineChart(cvs, ctx2, series, { showCursor: true });
  }

  /* ═══════════════════════════════════════════════
     VIEW 5: PHYSICAL REALITY CHECK (ODE DRIFT)
  ═══════════════════════════════════════════════ */
  function renderReality() {
    const cvs = document.getElementById("chart-drift");
    if (!cvs) return;
    const ctx2 = cvs.getContext("2d");
    const series = [];

    for (const [key, traj] of Object.entries(activeScenario.trajectories)) {
      if (!traj.reality_check) continue;
      const rc = traj.reality_check;
      series.push({
        x: rc.t_sim,
        y: rc.ee_drift,
        color: PALETTE[key] || PALETTE.text,
        lineWidth: 2.2,
      });
    }
    drawLineChart(cvs, ctx2, series, { showCursor: false, yMin: 0 });
  }

  /* ═══════════════════════════════════════════════
     DOCUMENT LOAD
  ═══════════════════════════════════════════════ */
  document.addEventListener("DOMContentLoaded", init);
})();
