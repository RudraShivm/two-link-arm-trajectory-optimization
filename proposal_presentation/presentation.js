// SUPERSEDED. This deck mixed the CSE thesis (ManiSkill / VLM) with
// AMCS 2025 BIP/sparsity. The course project is a 2-link arm + three
// numerical methods. Present presentation.pdf instead.
//
const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5

// ---- Palette (light theme, engineering / robotics feel) ----
const NAVY = "16324F";      // deep slate navy - headers / dominant text
const TEAL = "2E86AB";      // accent teal
const TEAL_DK = "1B5E75";
const BG = "FFFFFF";        // white background
const CARD = "F2F6F8";      // very light cool gray card
const MUTE = "5B7182";      // muted body text
const LINE = "D8E2E8";

const FIG_DIR = "/home/claude/figs";

const TITLE_OPT = { fontFace: "Cambria", color: NAVY, bold: true };
const BODY_OPT = { fontFace: "Calibri", color: MUTE };

function pageNum(slide, n, light) {
  slide.addText(`0${n} / 05`, {
    x: 12.0, y: 7.12, w: 0.9, h: 0.3,
    fontFace: "Calibri", fontSize: 9, color: light ? "8FD6E8" : MUTE, align: "right",
  });
}

function kicker(slide, text) {
  slide.addText(text.toUpperCase(), {
    x: 0.6, y: 0.42, w: 8, h: 0.3,
    fontFace: "Calibri", fontSize: 12, color: TEAL, bold: true, charSpacing: 2,
  });
}

/* =========================================================
   SLIDE 1 — Title
   ========================================================= */
{
  const s = pres.addSlide();
  s.background = { color: BG };

  // soft geometric motif: rounded rect block on right, echoing sparsity "blocks"
  s.addShape("roundRect", { x: 9.6, y: 0, w: 3.7, h: 7.5, fill: { color: CARD }, line: { type: "none" }, rectRadius: 0 });

  // block-diagonal motif dots (echoes the paper's sparsity pattern)
  const dotColors = [TEAL, NAVY, TEAL_DK];
  for (let i = 0; i < 6; i++) {
    s.addShape("roundRect", {
      x: 9.85 + i * 0.36, y: 1.0 + i * 0.85, w: 1.35, h: 0.62,
      fill: { color: "FFFFFF" }, line: { color: LINE, width: 1 }, rectRadius: 0.08,
    });
    s.addShape("ellipse", {
      x: 10.0 + i * 0.36, y: 1.18 + i * 0.85, w: 0.16, h: 0.16,
      fill: { color: dotColors[i % 3] }, line: { type: "none" },
    });
    s.addShape("ellipse", {
      x: 10.25 + i * 0.36, y: 1.32 + i * 0.85, w: 0.16, h: 0.16,
      fill: { color: dotColors[(i + 1) % 3] }, line: { type: "none" },
    });
  }

  s.addText("PROJECT PROPOSAL", {
    x: 0.7, y: 1.55, w: 6, h: 0.35,
    fontFace: "Calibri", fontSize: 13, color: TEAL, bold: true, charSpacing: 3,
  });

  s.addText("Predictable Computation for\nHighly-Articulated & Embodied Systems", {
    x: 0.7, y: 2.0, w: 8.4, h: 1.9,
    fontFace: "Cambria", fontSize: 34, color: NAVY, bold: true, lineSpacingMultiple: 1.08,
  });

  s.addText(
    "Grounded in sparsity-free trajectory optimization for robots with many degrees of freedom, and extended toward real-time, predictable inference for embodied world models.",
    { x: 0.7, y: 3.95, w: 8.2, h: 0.95, fontFace: "Calibri", fontSize: 14, color: MUTE, lineSpacingMultiple: 1.25 }
  );

  s.addShape("line", { x: 0.7, y: 5.15, w: 1.1, h: 0, line: { color: TEAL, width: 2.5 } });

  s.addText(
    [
      { text: "Reference paper: ", options: { bold: true, color: NAVY } },
      { text: "Cardona-Ortiz, D. & Arechavaleta, G. (2025). Trajectory Optimization for Highly Articulated Robots Based on Sparsity-Free Local Direct Collocation. Int. J. Appl. Math. Comput. Sci., 35(4), 577–589.", options: { color: MUTE } },
    ],
    { x: 0.7, y: 5.4, w: 8.2, h: 0.8, fontFace: "Calibri", fontSize: 11, lineSpacingMultiple: 1.2 }
  );

  s.addText("Dibbo Chowdhury  ·  BUET, Dept. of CSE", {
    x: 0.7, y: 6.7, w: 6, h: 0.35, fontFace: "Calibri", fontSize: 12, color: NAVY, bold: true,
  });
  pageNum(s, 1);
}

/* =========================================================
   SLIDE 2 — The Problem
   ========================================================= */
{
  const s = pres.addSlide();
  s.background = { color: BG };
  kicker(s, "01 · The Problem");
  s.addText("Generic optimal-control solvers don't scale to many-DoF robots", {
    x: 0.6, y: 0.75, w: 8.6, h: 0.9, ...TITLE_OPT, fontSize: 25, lineSpacingMultiple: 1.05,
  });

  const bullets = [
    { text: "Robot trajectory optimization is transcribed via direct collocation into a large nonlinear program: decision vector z stacks state x(t) and control u(t) at every collocation point.", options: { bullet: { code: "25AA" }, color: MUTE, breakLine: true } },
    { text: "Solvers need the constraint Jacobian J(z) = A + B·D(z) at every iteration — and D(z) grows with the robot's degrees of freedom (DoFs).", options: { bullet: { code: "25AA" }, color: MUTE, breakLine: true } },
    { text: "General-purpose frameworks (CasADi) use automatic differentiation over dense data structures — memory and compute blow up for humanoids and mobile manipulators.", options: { bullet: { code: "25AA" }, color: MUTE, breakLine: false } },
  ];
  s.addText(bullets, {
    x: 0.6, y: 1.85, w: 6.6, h: 3.1, fontFace: "Calibri", fontSize: 13.5, lineSpacingMultiple: 1.32, paraSpaceAfter: 12,
  });

  // stat callout card
  s.addShape("roundRect", { x: 0.6, y: 5.15, w: 6.6, h: 1.55, fill: { color: CARD }, line: { color: LINE, width: 1 }, rectRadius: 0.08 });
  s.addText("J(z)  ∈  ℝ^(nc × nz)", { x: 0.9, y: 5.3, w: 3, h: 0.5, fontFace: "Cambria", fontSize: 20, color: NAVY, bold: true });
  s.addText("A highly sparse matrix — yet generic AD tools evaluate and store it as if it were dense.", {
    x: 0.9, y: 5.85, w: 5.9, h: 0.75, fontFace: "Calibri", fontSize: 12, color: MUTE, italic: true, lineSpacingMultiple: 1.2,
  });

  // Figure: sparsity pattern
  s.addShape("roundRect", { x: 7.55, y: 1.85, w: 5.15, h: 4.9, fill: { color: "FFFFFF" }, line: { color: LINE, width: 1 }, rectRadius: 0.06 });
  s.addImage({ path: path.join(FIG_DIR, "fig1_clean.png"), x: 7.8, y: 2.05, w: 4.65, h: 1.88 });
  s.addText("Fig. 1 (Cardona-Ortiz & Arechavaleta, 2025) — block-diagonal sparsity of the defect-constraint matrices A and B under trapezoidal collocation.", {
    x: 7.8, y: 4.0, w: 4.65, h: 0.95, fontFace: "Calibri", fontSize: 10.5, color: MUTE, italic: true, lineSpacingMultiple: 1.25,
  });
  s.addText("Every block Aₖ, Bₖ repeats identically — the sparsity pattern is known before the solver ever runs.", {
    x: 7.8, y: 5.15, w: 4.65, h: 1.4, fontFace: "Calibri", fontSize: 12.5, color: NAVY, bold: true, lineSpacingMultiple: 1.3,
  });
  pageNum(s, 2);
}

/* =========================================================
   SLIDE 3 — The Paper's Method (BIP)
   ========================================================= */
{
  const s = pres.addSlide();
  s.background = { color: BG };
  kicker(s, "02 · The Method");
  s.addText("Block Indexation Procedure: exploit known structure instead of rediscovering it", {
    x: 0.6, y: 0.75, w: 12, h: 0.9, ...TITLE_OPT, fontSize: 23, lineSpacingMultiple: 1.05,
  });

  // Left: figure (performance chart)
  s.addShape("roundRect", { x: 0.6, y: 1.8, w: 4.05, h: 4.9, fill: { color: "FFFFFF" }, line: { color: LINE, width: 1 }, rectRadius: 0.06 });
  s.addImage({ path: path.join(FIG_DIR, "fig3_tight.png"), x: 0.8, y: 1.98, w: 3.65, h: 3.88 });
  s.addText("Fig. 3 — constraint-Jacobian time vs. DoF. BIP-A (bottom curve, all 3 panels) beats sparse finite differences and CasADi's automatic differentiation at every problem size.", {
    x: 0.8, y: 5.9, w: 3.65, h: 0.75, fontFace: "Calibri", fontSize: 9.5, color: MUTE, italic: true, lineSpacingMultiple: 1.2,
  });

  // Right: 3 step cards
  const steps = [
    { n: "1", h: "Predict the sparsity", d: "The pattern of A, B and D(z) is deducible in advance from Nc, ns and nu — no need to detect it at runtime." },
    { n: "2", h: "Store only triples", d: "Each nonzero is kept as (row, col, value). One block is generated once, then shifted and stacked — never rebuilt from scratch." },
    { n: "3", h: "Sparse assembly only", d: "T_J = T_A ⊕ T_B ⊗ T_D. Sparse–sparse addition and multiplication replace dense matrix algebra entirely." },
  ];
  let y = 1.8;
  steps.forEach((st) => {
    s.addShape("ellipse", { x: 5.1, y: y + 0.02, w: 0.5, h: 0.5, fill: { color: TEAL }, line: { type: "none" } });
    s.addText(st.n, { x: 5.1, y: y + 0.02, w: 0.5, h: 0.5, align: "center", valign: "middle", fontFace: "Cambria", fontSize: 18, color: "FFFFFF", bold: true });
    s.addText(st.h, { x: 5.8, y: y - 0.05, w: 6.9, h: 0.4, fontFace: "Calibri", fontSize: 15, color: NAVY, bold: true });
    s.addText(st.d, { x: 5.8, y: y + 0.35, w: 6.9, h: 0.75, fontFace: "Calibri", fontSize: 12, color: MUTE, lineSpacingMultiple: 1.25 });
    y += 1.55;
  });

  s.addShape("roundRect", { x: 5.1, y: 6.55, w: 7.6, h: 0.68, fill: { color: CARD }, line: { color: LINE, width: 1 }, rectRadius: 0.08 });
  s.addText("Paired with Pinocchio's analytical dynamics derivatives, this gives BIP-A — ~4× faster than automatic differentiation, and the only method that scales past 36 DoF.", {
    x: 5.3, y: 6.55, w: 7.2, h: 0.68, fontFace: "Calibri", fontSize: 11.5, color: NAVY, valign: "middle", lineSpacingMultiple: 1.2,
  });
  pageNum(s, 3);
}

/* =========================================================
   SLIDE 4 — Validated on Real Robots
   ========================================================= */
{
  const s = pres.addSlide();
  s.background = { color: BG };
  kicker(s, "03 · Validation");
  s.addText("The result holds on real, high-DoF articulated platforms", {
    x: 0.6, y: 0.75, w: 10.5, h: 0.9, ...TITLE_OPT, fontSize: 25,
  });

  s.addShape("roundRect", { x: 0.6, y: 1.75, w: 12.1, h: 2.55, fill: { color: "FFFFFF" }, line: { color: LINE, width: 1 }, rectRadius: 0.06 });
  s.addImage({ path: path.join(FIG_DIR, "fig4_final.png"), x: 0.85, y: 1.95, w: 11.6, h: 2.15 });
  s.addText("Fig. 4 — NAO humanoid (24 DoF) generated via the proposed NOCS: arm motion while balanced on one leg (top), and an airplane-like posture (bottom), both under a centroidal-momentum path constraint.", {
    x: 0.85, y: 4.1, w: 11.6, h: 0.35, fontFace: "Calibri", fontSize: 10, color: MUTE, italic: true,
  });

  // Stat row
  const stats = [
    { big: "6 → 24", small: "DoFs tested\nUR5 · ABB YuMi · NAO" },
    { big: "≥50%", small: "less compute time\nthan CasADi, every case" },
    { big: "100", small: "DoF snake-robot\nADA cannot even run" },
    { big: "4×", small: "faster Jacobian eval.\nBIP-A vs. automatic diff." },
  ];
  const cardW = 2.9, gap = 0.18;
  stats.forEach((st, i) => {
    const x = 0.6 + i * (cardW + gap);
    s.addShape("roundRect", { x, y: 4.75, w: cardW, h: 1.95, fill: { color: CARD }, line: { color: LINE, width: 1 }, rectRadius: 0.08 });
    s.addText(st.big, { x, y: 4.9, w: cardW, h: 0.75, align: "center", fontFace: "Cambria", fontSize: 30, color: TEAL_DK, bold: true });
    s.addText(st.small, { x, y: 5.65, w: cardW, h: 0.9, align: "center", fontFace: "Calibri", fontSize: 11, color: NAVY, lineSpacingMultiple: 1.2 });
  });
  pageNum(s, 4);
}

/* =========================================================
   SLIDE 5 — Why this grounds our project
   ========================================================= */
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("04 · WHY IT MATTERS FOR US".toUpperCase(), {
    x: 0.6, y: 0.42, w: 8, h: 0.3, fontFace: "Calibri", fontSize: 12, color: "8FD6E8", bold: true, charSpacing: 2,
  });
  s.addText("From predictable robot dynamics to predictable embodied inference", {
    x: 0.6, y: 0.75, w: 12.1, h: 1.0, fontFace: "Cambria", fontSize: 25, color: "FFFFFF", bold: true, lineSpacingMultiple: 1.05,
  });

  s.addText(
    "The paper's core insight: when a system's structure is knowable in advance (here, sparsity from Nc, ns, nu), computation can be made both faster and bounded — not just fast on average. Our project applies the same principle to a different high-dimensional, real-time problem.",
    { x: 0.6, y: 1.85, w: 12.1, h: 0.85, fontFace: "Calibri", fontSize: 13, color: "CADCFC", lineSpacingMultiple: 1.3 }
  );

  const cols = [
    {
      title: "Their problem",
      body: "Evaluate a huge, structured Jacobian for many-DoF robot dynamics inside every NLP iteration — dense AD tools don't scale.",
    },
    {
      title: "Their fix",
      body: "Precompute the known sparsity pattern once (BIP); do only sparse-sparse algebra at solve time — deterministic, low-memory, ~4× faster.",
    },
    {
      title: "Our project",
      body: "\u201CPredictable VLM/VLA on Multicore+GPU\u201D: schedule embodied world-model inference under a latency budget using an adaptive executor that exploits known model/hardware structure, tracked via a belief-age metric, evaluated on ManiSkill.",
    },
  ];
  const colW = 3.93, colGap = 0.15;
  cols.forEach((c, i) => {
    const x = 0.6 + i * (colW + colGap);
    s.addShape("roundRect", { x, y: 2.95, w: colW, h: 3.55, fill: { color: i === 2 ? TEAL : "1F4A63" }, line: { type: "none" }, rectRadius: 0.08 });
    s.addText(c.title, { x: x + 0.28, y: 3.2, w: colW - 0.56, h: 0.5, fontFace: "Cambria", fontSize: 16, color: "FFFFFF", bold: true });
    s.addText(c.body, { x: x + 0.28, y: 3.75, w: colW - 0.56, h: 2.6, fontFace: "Calibri", fontSize: 12, color: "EAF3F7", lineSpacingMultiple: 1.32 });
  });

  s.addText(
    "Takeaway: structure-aware, sparsity/predictability-first computation — proven here for classical trajectory optimization — is the design principle we carry into real-time embodied ML inference.",
    { x: 0.6, y: 6.75, w: 12.1, h: 0.55, fontFace: "Calibri", fontSize: 12, color: "8FD6E8", italic: true, bold: true, lineSpacingMultiple: 1.2 }
  );
  pageNum(s, 5, true);
}

pres.writeFile({ fileName: "/home/claude/deck_out.pptx" }).then(() => {
  console.log("written");
});