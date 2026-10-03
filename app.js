const round = (value, places = 3) => Number(value.toFixed(places));
const dot = (left, right) => left.reduce((sum, value, index) => sum + value * right[index], 0);
const add = (left, right) => left.map((value, index) => value + right[index]);
const scale = (vector, scalar) => vector.map(value => value * scalar);

export function stableSoftmax(scores) {
  const maximum = Math.max(...scores);
  const exponentials = scores.map(score => Math.exp(score - maximum));
  const normalizer = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map(value => value / normalizer);
}

export function standardAttention(q, k, v) {
  const scaleFactor = 1 / Math.sqrt(q[0].length);
  const scores = q.map(query => k.map(key => dot(query, key) * scaleFactor));
  const probabilities = scores.map(stableSoftmax);
  const output = probabilities.map(row => row.reduce(
    (accumulator, weight, index) => add(accumulator, scale(v[index], weight)),
    Array(v[0].length).fill(0),
  ));
  return { scores, probabilities, output };
}

export function streamingAttention(q, k, v, blockSize) {
  const scaleFactor = 1 / Math.sqrt(q[0].length);
  const output = [];
  const states = [];
  for (const [row, query] of q.entries()) {
    let maximum = -Infinity;
    let normalizer = 0;
    let numerator = Array(v[0].length).fill(0);
    const rowStates = [];
    for (let start = 0; start < k.length; start += blockSize) {
      const blockScores = k.slice(start, start + blockSize).map(key => dot(query, key) * scaleFactor);
      const nextMaximum = Math.max(maximum, ...blockScores);
      const oldScale = Number.isFinite(maximum) ? Math.exp(maximum - nextMaximum) : 0;
      const weights = blockScores.map(score => Math.exp(score - nextMaximum));
      normalizer = oldScale * normalizer + weights.reduce((sum, weight) => sum + weight, 0);
      numerator = add(scale(numerator, oldScale), weights.reduce(
        (accumulator, weight, index) => add(accumulator, scale(v[start + index], weight)),
        Array(v[0].length).fill(0),
      ));
      maximum = nextMaximum;
      rowStates.push({ block: start / blockSize + 1, maximum, normalizer, numerator: [...numerator], output: scale(numerator, 1 / normalizer) });
    }
    output.push(scale(numerator, 1 / normalizer));
    states.push({ row, blocks: rowStates });
  }
  return { output, states };
}

function matrixText(matrix) {
  return JSON.stringify(matrix.map(row => row.map(value => round(value))));
}

function initializeAttentionCalculator() {
  const root = document.querySelector("#attention-calc");
  if (!root) return;
  const q = [[1, 0], [0, 1]];
  const k = [[1, 0], [0, 1], [1, 1]];
  const v = [[2], [3], [5]];
  const result = standardAttention(q, k, v);

  const scoreEl = document.querySelector("#score-output");
  const stage1 = document.querySelector("#att-stage1");
  const stage2 = document.querySelector("#att-stage2");
  const stage3 = document.querySelector("#attention-output");
  const card = document.querySelector("#att-step-card");
  const cardLabel = document.querySelector("#att-step-label");
  const indicator = document.querySelector("#att-step-indicator");
  const prev = document.querySelector("#att-prev");
  const next = document.querySelector("#att-next");
  const status = document.querySelector("#attention-status");
  if (!scoreEl || !stage1 || !stage2 || !stage3 || !card || !prev || !next) return;

  const steps = [
    { label: "scores", value: result.scores, cardStage: "stage-1" },
    { label: "P = softmax(S)", value: result.probabilities, cardStage: "stage-2" },
    { label: "output O = PV", value: result.output, cardStage: "stage-3" },
  ];
  let current = 0;

  const render = () => {
    const step = steps[current];
    scoreEl.textContent = matrixText(step.value);
    cardLabel.textContent = step.label;
    card.classList.remove("stage-1", "stage-2", "stage-3");
    card.classList.add(step.cardStage);
    stage1.textContent = matrixText(result.scores);
    stage2.textContent = matrixText(result.probabilities);
    stage3.textContent = matrixText(result.output);
    indicator.textContent = `step ${current + 1} / ${steps.length}`;
    prev.disabled = current === 0;
    next.disabled = current === steps.length - 1;
    status.textContent = current === 0
      ? "Step 1 of 3: scores. S = QK^T / √d. Naive attention writes this N×N matrix to HBM and reads it back for softmax — the source of the memory wall."
      : current === 1
        ? "Step 2 of 3: row-softmax. Subtract each row's max, exponentiate, divide by the sum. Rows now sum to 1; entries are probabilities P."
        : "Step 3 of 3: output. O = PV is a weighted average of value rows. Naive attention writes P back to HBM first; FA-2 keeps it on chip.";
  };

  prev.addEventListener("click", () => { if (current > 0) { current--; render(); } });
  next.addEventListener("click", () => { if (current < steps.length - 1) { current++; render(); } });
  root.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") { event.preventDefault(); prev.click(); }
    else if (event.key === "ArrowRight") { event.preventDefault(); next.click(); }
  });
  render();
}

const STATE_DEMOS = {
  "same-max": { q: [[1, 0]], k: [[1, 0], [0, 1], [1, 1]], v: [[2], [3], [5]] },
  "new-max": { q: [[0, 5]], k: [[5, 0], [1, 0], [0, 1]], v: [[2], [3], [5]] },
};

function computeScores(q, k) {
  const scale = 1 / Math.sqrt(q[0].length);
  return k.map(key => round(q[0].reduce((sum, value, index) => sum + value * key[index], 0) * scale, 3));
}

function initializeStreamingState() {
  const fields = {
    tile: document.querySelector("#state-tile"),
    maximum: document.querySelector("#state-max"),
    normalizer: document.querySelector("#state-norm"),
    numerator: document.querySelector("#state-numerator"),
    output: document.querySelector("#state-output"),
    oldScale: document.querySelector("#state-old-scale"),
    newMass: document.querySelector("#state-new-mass"),
    mergeRule: document.querySelector("#state-merge-rule"),
    status: document.querySelector("#state-status"),
  };
  if (!fields.tile) return;
  const play = document.querySelector("#state-play");
  const scoreCellsContainer = document.querySelector("#state-cells");
  const progressFill = document.querySelector("#state-progress-fill");
  const progressText = document.querySelector("#state-progress-text");
  const presetButtons = [...document.querySelectorAll("[data-state-demo]")];

  let demo = [];
  let scores = [];
  let index = 0;
  let timer;
  const stopPlaying = () => {
    clearInterval(timer);
    timer = undefined;
    play.disabled = false;
    play.setAttribute("aria-pressed", "false");
    play.textContent = "Play";
  };
  const render = () => {
    const state = demo[index - 1];
    if (!state) {
      fields.tile.textContent = "—";
      fields.maximum.textContent = "−∞";
      fields.normalizer.textContent = "0";
      fields.numerator.textContent = "0";
      fields.output.textContent = "—";
      if (fields.oldScale) fields.oldScale.textContent = "—";
      if (fields.newMass) fields.newMass.textContent = "—";
      if (fields.mergeRule) fields.mergeRule.textContent = "waiting for a tile";
      progressText.textContent = `0 / ${demo.length}`;
      progressFill.style.width = "0%";
      for (const cell of scoreCellsContainer.children) { cell.classList.remove("active", "done"); }
      fields.status.textContent = "Start with an empty state. The final result matches full-row softmax.";
      return;
    }
    fields.tile.textContent = `${state.block} / ${demo.length}`;
    fields.maximum.textContent = round(state.maximum);
    fields.normalizer.textContent = round(state.normalizer);
    fields.numerator.textContent = round(state.numerator[0]);
    fields.output.textContent = round(state.output[0]);
    progressText.textContent = `${state.block} / ${demo.length}`;
    progressFill.style.width = `${(state.block / demo.length * 100).toFixed(1)}%`;
    const tileStart = (state.block - 1) * 2;
    for (const [i, cell] of [...scoreCellsContainer.children].entries()) {
      cell.classList.toggle("active", i >= tileStart && i < tileStart + 2 && index !== demo.length);
      cell.classList.toggle("done", i < tileStart + 2 && index === demo.length);
    }
    const previousState = index > 1 ? demo[index - 2] : undefined;
    const previousMax = previousState ? previousState.maximum : -Infinity;
    const rescales = Boolean(previousState && state.maximum > previousMax);
    const oldScale = previousState ? Math.exp(previousMax - state.maximum) : 0;
    const tileMass = scores.slice(tileStart, tileStart + 2).reduce((sum, score) => sum + Math.exp(score - state.maximum), 0);
    if (fields.oldScale) fields.oldScale.textContent = round(oldScale);
    if (fields.newMass) fields.newMass.textContent = round(tileMass);
    if (fields.mergeRule) fields.mergeRule.textContent = !previousState
      ? "initialize state"
      : rescales ? "rebase + add" : "add at same max";
    fields.status.textContent = index === demo.length
      ? `Complete: O = ${round(state.output[0])}, exactly the same value as a full-row stable softmax.`
      : rescales
        ? `Tile ${state.block}: a new maximum (${round(previousMax)} → ${round(state.maximum)}) forces the old ℓ and o to be rescaled by exp(m_old − m_new) before the new contribution lands.`
        : `Tile ${state.block} updates m, ℓ, and o. Existing state is rescaled only if a new maximum arrives.`;
  };
  const step = () => { index = Math.min(index + 1, demo.length); render(); };

  const loadDemo = (name) => {
    stopPlaying();
    const config = STATE_DEMOS[name] ?? STATE_DEMOS["same-max"];
    demo = streamingAttention(config.q, config.k, config.v, 2).states[0].blocks;
    scores = computeScores(config.q, config.k);
    scoreCellsContainer.replaceChildren();
    for (const [i, value] of scores.entries()) {
      const cell = document.createElement("span");
      cell.className = "state-cell";
      const label = document.createElement("span");
      label.textContent = `k_${i + 1}`;
      const num = document.createElement("strong");
      num.textContent = value.toString();
      cell.append(label, num);
      scoreCellsContainer.append(cell);
    }
    index = 0;
    render();
    for (const button of presetButtons) {
      const selected = button.dataset.stateDemo === name;
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    }
  };

  document.querySelector("#state-step").addEventListener("click", step);
  document.querySelector("#state-reset").addEventListener("click", () => { stopPlaying(); index = 0; render(); });
  play.addEventListener("click", () => {
    stopPlaying();
    index = 0;
    step();
    play.disabled = true;
    play.setAttribute("aria-pressed", "true");
    play.textContent = "Playing…";
    timer = setInterval(() => {
      step();
      if (index === demo.length) stopPlaying();
    }, 850);
  });
  for (const button of presetButtons) {
    button.addEventListener("click", () => loadDemo(button.dataset.stateDemo));
  }
  fields.tile.closest(".widget").addEventListener("keydown", (event) => {
    if (event.target.tagName === "INPUT" || event.target.tagName === "TEXTAREA") return;
    if (event.key === " " || event.key === "ArrowRight") { event.preventDefault(); step(); }
    else if (event.key === "r" || event.key === "R") { event.preventDefault(); document.querySelector("#state-reset").click(); }
  });
  loadDemo("same-max");
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${round(bytes, 0)}`;
  if (bytes < 1024 ** 2) return `${round(bytes / 1024, 1)} Ki`;
  if (bytes < 1024 ** 3) return `${round(bytes / 1024 ** 2, 1)} Mi`;
  return `${round(bytes / 1024 ** 3, 2)} Gi`;
}

function formatNumber(value) {
  if (value < 1000) return round(value, 0).toString();
  return value.toLocaleString("en-US").replace(/,/g, " ");
}

export const _internals = { formatBytes, formatNumber };

// --- Hero: the forward pass, drawn ------------------------------------------
//
// The whole guide is one idea: you never write the N x N score matrix. You keep
// one K/V tile on chip, walk it across the query rows, and fold every tile into
// three running numbers. So the hero draws that, once, on a loop.
//
// Six query rows, seven tiles. The tile is the only thing that moves; the rows
// behind it have already been folded in and never need revisiting. That is the
// whole trick, so the picture shows nothing else.

const SWEEP_ROWS = 6;
const SWEEP_TILES = 7;
const SWEEP_STEP_MS = 420;
const SWEEP_HOLD_MS = 1900;

function initializeHeroSweep() {
  const canvas = document.querySelector("#hero-sweep-canvas");
  if (!canvas) return;
  const context = canvas.getContext("2d");
  if (!context) return;
  const countEl = document.querySelector("#hero-sweep-count");
  const phaseEl = document.querySelector("#hero-sweep-phase");

  const styles = getComputedStyle(document.documentElement);
  const paper = styles.getPropertyValue("--paper-1").trim() || "#15191F";
  const rule = styles.getPropertyValue("--rule").trim() || "#2A2E34";
  const ink3 = styles.getPropertyValue("--ink-3").trim() || "#6E6A5F";
  const vermilion = styles.getPropertyValue("--vermilion").trim() || "#C8553D";

  // How far along each row is, once `folded` tiles have landed. Deliberately
  // uneven: rows converge at different rates, and a row that stops early is
  // what "the output is already stable" looks like.
  const CONVERGE = [0.94, 0.72, 0.86, 0.55, 0.78, 0.63];

  const prefersStill = window.matchMedia("(prefers-reduced-motion: reduce)");

  const draw = (folded, active, progress) => {
    const w = canvas.width;
    const h = canvas.height;
    const padX = 18;
    const padTop = 16;
    const labelW = 30;
    const gapX = 3;
    const gapY = 4;
    const cols = SWEEP_TILES;
    const rows = SWEEP_ROWS;
    const cellW = (w - padX * 2 - labelW) / cols;
    const cellH = (h - padTop * 2) / rows;

    context.clearRect(0, 0, w, h);

    for (let row = 0; row < rows; row++) {
      const y = padTop + row * cellH;
      context.fillStyle = ink3;
      context.globalAlpha = 0.9;
      context.font = "11px ui-monospace, monospace";
      context.textAlign = "right";
      context.textBaseline = "middle";
      context.fillText(`q${row + 1}`, labelW + padX - 8, y + cellH / 2);
      context.globalAlpha = 1;

      for (let col = 0; col < cols; col++) {
        const x = padX + labelW + col * cellW;
        const box = [x + gapX / 2, y + gapY / 2, cellW - gapX, cellH - gapY];

        // Not reached yet: an empty slot waiting for its tile.
        if (col > active) {
          context.strokeStyle = rule;
          context.lineWidth = 1;
          context.strokeRect(box[0] + 0.5, box[1] + 0.5, box[2] - 1, box[3] - 1);
          continue;
        }

        // Already folded into the running sum.
        const share = Math.min(1, CONVERGE[row] * ((col + 1) / cols));
        context.fillStyle = vermilion;
        context.globalAlpha = 0.18 + 0.5 * share;
        context.fillRect(box[0], box[1], box[2], box[3]);
        context.globalAlpha = 1;

        // The tile on chip right now, mid-sweep.
        if (col === active) {
          context.fillStyle = vermilion;
          context.globalAlpha = 0.9;
          context.fillRect(box[0], box[1], box[2] * progress, box[3]);
          context.globalAlpha = 1;
        }
        context.strokeStyle = rule;
        context.lineWidth = 1;
        context.strokeRect(box[0] + 0.5, box[1] + 0.5, box[2] - 1, box[3] - 1);
      }
    }

    // The on-chip tile, drawn as the single shared block it is.
    const tileX = padX + labelW + active * cellW;
    context.strokeStyle = vermilion;
    context.lineWidth = 1.5;
    context.strokeRect(tileX + 1, padTop + 1, cellW - 2, (h - padTop * 2) - 2);
  };

  let frame = 0;
  let folded = 0;
  let startedAt = performance.now();

  const paint = () => {
    if (prefersStill.matches) {
      // One finished frame. The tile has been all the way across and every row
      // holds its answer; nothing moves.
      draw(SWEEP_TILES, SWEEP_TILES - 1, 1);
      if (countEl) countEl.textContent = `${SWEEP_TILES} / ${SWEEP_TILES} tiles`;
      if (phaseEl) phaseEl.textContent = "every row holds its answer";
      return;
    }
    const elapsed = performance.now() - startedAt;
    const perTile = SWEEP_STEP_MS;
    const cycle = perTile * SWEEP_TILES + SWEEP_HOLD_MS;
    const now = elapsed % cycle;
    const running = now < perTile * SWEEP_TILES;
    const slot = Math.min(SWEEP_TILES - 1, Math.floor(now / perTile));
    const progress = running ? (now % perTile) / perTile : 1;

    draw(Math.min(folded, slot + 1), slot, progress);
    folded = slot + 1;

    if (countEl) countEl.textContent = `${folded} / ${SWEEP_TILES} tiles`;
    if (phaseEl) {
      phaseEl.textContent = running
        ? "one K/V tile on chip"
        : "no tile in flight, nothing written to HBM";
    }
    frame = requestAnimationFrame(paint);
  };

  const fit = () => {
    const width = Math.max(260, Math.round(canvas.getBoundingClientRect().width));
    // 7 columns by 6 rows, so this ratio keeps the tiles square rather than
    // turning the grid into tall stripes.
    const ratio = SWEEP_TILES / SWEEP_ROWS;
    if (canvas.width === width) return;
    canvas.width = width;
    canvas.height = Math.round(width * ratio);
  };

  fit();
  new ResizeObserver(fit).observe(canvas);
  prefersStill.addEventListener("change", () => {
    cancelAnimationFrame(frame);
    startedAt = performance.now();
    paint();
  });
  document.addEventListener("visibilitychange", () => {
    cancelAnimationFrame(frame);
    // Coming back from a hidden tab would otherwise jump the sweep forward by
    // however long you were away.
    startedAt = performance.now();
    if (!document.hidden) paint();
  });
  paint();
}

function initializeTileObservatory() {
  const root = document.querySelector("#tile-size");
  if (!root) return;
  const nInput = document.querySelector("#sequence-length");
  const nValue = document.querySelector("#sequence-value");
  const dInput = document.querySelector("#head-dim");
  const dValue = document.querySelector("#head-dim-value");
  const bInput = document.querySelector("#tile-size-input");
  const bValue = document.querySelector("#tile-size-value");
  const status = document.querySelector("#tile-status");
  const barNaive = document.querySelector("#bar-naive");
  const barFA = document.querySelector("#bar-fa");
  const naiveTotal = document.querySelector("#naive-total");
  const faTotal = document.querySelector("#fa-total");
  const intensityNaive = document.querySelector("#intensity-naive");
  const intensityFA = document.querySelector("#intensity-fa");
  const ratio = document.querySelector("#traffic-ratio");
  if (!nInput || !dInput || !bInput || !barNaive || !barFA) return;

  const BYTES_PER_ELEMENT = 2;

  const render = () => {
    const n = Number(nInput.value);
    const d = Number(dInput.value);
    const blockRows = Number(bInput.value);
    nValue.textContent = formatNumber(n);
    dValue.textContent = formatNumber(d);
    bValue.textContent = formatNumber(blockRows);

    const qkvBytes = 3 * n * d * BYTES_PER_ELEMENT;
    const intermediateBytes = 4 * n * n * BYTES_PER_ELEMENT;
    const outputBytes = n * d * BYTES_PER_ELEMENT;
    const statisticsBytes = 2 * n * 4;

    const naiveTotalBytes = qkvBytes + intermediateBytes + outputBytes;
    const faTotalBytes = qkvBytes + outputBytes + statisticsBytes;

    const naiveLoad = qkvBytes;
    const naiveInter = intermediateBytes;
    const naiveOut = outputBytes;
    const faLoad = qkvBytes;
    const faOut = outputBytes + statisticsBytes;

    const renderBar = (bar, segments) => {
      bar.replaceChildren();
      const total = segments.reduce((sum, seg) => sum + seg.value, 0);
      for (const seg of segments) {
        const span = document.createElement("span");
        span.className = seg.className;
        span.style.width = total > 0 ? `${(seg.value / total * 100).toFixed(2)}%` : "0%";
        span.title = `${seg.label}: ${formatBytes(seg.value)} bytes`;
        bar.append(span);
      }
    };

    renderBar(barNaive, [
      { className: "segment-load", value: naiveLoad, label: "Q/K/V load" },
      { className: "segment-inter", value: naiveInter, label: "S, P intermediates" },
      { className: "segment-out", value: naiveOut, label: "O write" },
    ]);
    renderBar(barFA, [
      { className: "segment-load", value: faLoad, label: "Q/K/V load" },
      { className: "segment-out", value: faOut, label: "O write + (m, ℓ)" },
    ]);

    naiveTotal.textContent = formatBytes(naiveTotalBytes);
    faTotal.textContent = formatBytes(faTotalBytes);

    const flops = 4 * n * n * d;
    const naiveIntensity = naiveTotalBytes > 0 ? flops / naiveTotalBytes : 0;
    const faIntensity = faTotalBytes > 0 ? flops / faTotalBytes : 0;
    intensityNaive.textContent = `${round(naiveIntensity, 0)} FLOP/B`;
    intensityFA.classList.add("fa");
    intensityFA.textContent = `${formatNumber(round(faIntensity, 0))} FLOP/B`;

    const trafficRatio = faTotalBytes > 0 ? naiveTotalBytes / faTotalBytes : 0;
    ratio.textContent = `${formatNumber(round(trafficRatio, 0))}×`;

    const naiveOrder = naiveTotalBytes > 1024 ** 3 ? "GiB" : naiveTotalBytes > 1024 ** 2 ? "MiB" : "KiB";
    const faOrder = faTotalBytes > 1024 ** 3 ? "GiB" : faTotalBytes > 1024 ** 2 ? "MiB" : "KiB";
    const regime = naiveIntensity < d * 4 ? "memory-bound" : "near compute-bound";
    status.textContent =
      `At N = ${formatNumber(n)}, d = ${d}, B_r = ${blockRows} in FP16/BF16: naive ≈ ${formatBytes(naiveTotalBytes)} (${naiveOrder}) per head, FA-2 ≈ ${formatBytes(faTotalBytes)} (${faOrder}) per head. ` +
      `Arithmetic intensity rises from ${round(naiveIntensity, 0)} to ${round(faIntensity, 0)} FLOP/B — a ${round(trafficRatio, 0)}× reduction in HBM traffic. The naive kernel is ${regime}; FA-2 is firmly compute-bound for any reasonable d.`;
  };

  for (const input of [nInput, dInput, bInput]) {
    input.addEventListener("input", render);
  }
  render();
}

function initializeWorkMap() {
  const diagram = document.querySelector("#work-diagram");
  const status = document.querySelector("#work-status");
  const buttons = [...document.querySelectorAll("[data-work-mode]")];
  const outputStat = document.querySelector("#work-output-stat");
  const sharedStat = document.querySelector("#work-shared-stat");
  const trafficStat = document.querySelector("#work-traffic-stat");
  if (!diagram || !status) return;

  const render = mode => {
    const isFA2 = mode === "fa2";
    diagram.replaceChildren();
    for (let i = 0; i < 4; i++) {
      const cell = document.createElement("div");
      cell.className = `work-cell ${mode}`;
      const title = document.createElement("span");
      title.className = "work-cell-title";
      title.textContent = `warp ${i}`;
      const body = document.createElement("span");
      body.className = "work-cell-body";
      if (isFA2) {
        body.textContent = `Q rows ${i * 16}–${(i + 1) * 16 - 1}\nshared K/V tile`;
      } else {
        body.textContent = `K/V cols ${i * 16}–${(i + 1) * 16 - 1}\npartial output`;
      }
      cell.append(title, body);
      diagram.append(cell);
    }
    if (outputStat) outputStat.textContent = isFA2 ? "independent per warp" : "partial per warp";
    if (sharedStat) sharedStat.textContent = isFA2 ? "one shared K/V tile" : "distinct K/V per warp";
    if (trafficStat) trafficStat.textContent = isFA2 ? "none" : "yes (shared-mem reduce)";
    status.textContent = isFA2
      ? "FA-2 sliced-Q: each warp owns independent Q output rows while sharing one K/V tile in shared memory. No cross-warp reduction."
      : "FA-1 sliced-K: warps take distinct K/V slices, produce partial output contributions, then need a cross-warp reduction through shared memory.";
    buttons.forEach(button => { const selected = button.dataset.workMode === mode; button.classList.toggle("selected", selected); button.setAttribute("aria-pressed", String(selected)); });
  };
  buttons.forEach(button => button.addEventListener("click", () => render(button.dataset.workMode)));
  render("fa1");
}

function initializeProgressRail() {
  const links = [...document.querySelectorAll(".rail nav a")];
  const sections = links.map(link => document.querySelector(link.getAttribute("href"))).filter(Boolean);
  const progressText = document.querySelector("#rail-progress-text");
  const progressFill = document.querySelector("#rail-progress-fill");
  const resumeLink = document.querySelector("#resume-reading");
  const storageKey = "fa2-guide-last-chapter";
  const render = (section, { persist = true } = {}) => {
    const index = sections.indexOf(section);
    if (index < 0) return;
    const chapter = index + 1;
    links.forEach(link => {
      const active = link.getAttribute("href") === `#${section.id}`;
      link.classList.toggle("active", active);
      link.toggleAttribute("aria-current", active);
    });
    if (progressText) progressText.textContent = `Chapter ${String(chapter).padStart(2, "0")} / ${String(sections.length).padStart(2, "0")}`;
    if (progressFill) progressFill.style.width = `${(chapter / sections.length * 100).toFixed(1)}%`;
    if (persist) {
      try { localStorage.setItem(storageKey, section.id); } catch { /* Reading progress is optional. */ }
    }
  };
  try {
    const savedId = localStorage.getItem(storageKey);
    const savedSection = sections.find(section => section.id === savedId);
    if (savedSection && savedSection.id !== sections[0]?.id && resumeLink) {
      resumeLink.href = `#${savedSection.id}`;
      resumeLink.firstChild.textContent = `Resume chapter ${String(sections.indexOf(savedSection) + 1).padStart(2, "0")} `;
      resumeLink.hidden = false;
    }
  } catch { /* Private browsing can deny local storage. */ }
  if (sections[0]) render(sections[0], { persist: false });
  let isInitialObservation = true;
  const observer = new IntersectionObserver(entries => {
    const visible = entries.filter(entry => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    render(visible.target, { persist: !isInitialObservation });
    isInitialObservation = false;
  }, { rootMargin: "-20% 0px -65% 0px", threshold: [0, .1, .5] });
  sections.forEach(section => observer.observe(section));
}

export function calculateNaiveBytes(n, d) {
  return 2 * (3 * n * d + 4 * n * n + n * d);
}

export function calculateFABytes(n, d) {
  return 2 * (3 * n * d + n * d) + 2 * n * 4;
}

export function calculateSpeedup(n, batchHeads) {
  const d = 128;
  const naiveBytes = calculateNaiveBytes(n, d);
  const faBytes = calculateFABytes(n, d);
  const ioRatio = naiveBytes / faBytes;
  const naiveIntensity = (4 * n * n * d) / naiveBytes;
  const ridgePoint = 208;
  const computeEfficiency = Math.min(1, naiveIntensity / ridgePoint);
  const occupancy = Math.min(1, Math.log2(Math.max(2, batchHeads)) / Math.log2(256));
  const speedup = ioRatio * computeEfficiency * (0.75 + 0.25 * occupancy);
  return Math.min(speedup, 5.0);
}

export function isSpeedupCapped(n, batchHeads) {
  const d = 128;
  const naiveBytes = calculateNaiveBytes(n, d);
  const faBytes = calculateFABytes(n, d);
  const ioRatio = naiveBytes / faBytes;
  const naiveIntensity = (4 * n * n * d) / naiveBytes;
  const ridgePoint = 208;
  const computeEfficiency = Math.min(1, naiveIntensity / ridgePoint);
  const occupancy = Math.min(1, Math.log2(Math.max(2, batchHeads)) / Math.log2(256));
  return ioRatio * computeEfficiency * (0.75 + 0.25 * occupancy) > 5.0;
}

function initializeSpeedupCalculator() {
  const root = document.querySelector("#speedup-calc");
  if (!root) return;
  const nInput = document.querySelector("#speedup-n-input");
  const bInput = document.querySelector("#speedup-b-input");
  const nDisplay = document.querySelector("#speedup-n");
  const bDisplay = document.querySelector("#speedup-b");
  const result = document.querySelector("#speedup-result");
  const regime = document.querySelector("#speedup-regime");
  const hbm = document.querySelector("#speedup-hbm");
  const barFill = document.querySelector("#speedup-bar-fill");
  const barMarker = document.querySelector("#speedup-bar-marker");
  const status = document.querySelector("#speedup-status");
  if (!nInput || !bInput || !barFill || !barMarker) return;

  const update = () => {
    const n = Number(nInput.value);
    const b = Number(bInput.value);
    const speedup = calculateSpeedup(n, b);
    const naive = calculateNaiveBytes(n, 128);
    const fa = calculateFABytes(n, 128);
    const ratio = naive / fa;
    nDisplay.textContent = n.toLocaleString("en-US").replace(/,/g, " ");
    bDisplay.textContent = b.toLocaleString("en-US");
    result.textContent = `${speedup.toFixed(1)}×`;
    if (regime) regime.textContent = speedup >= 3.0 ? "memory-bound" : speedup >= 2.0 ? "balanced" : "compute-bound";
    if (hbm) hbm.textContent = `≈ ${formatNumber(round(ratio, 0))}×`;
    const widthPct = Math.min(speedup / 6 * 100, 100);
    barFill.style.width = `${widthPct.toFixed(1)}%`;
    barMarker.style.left = `${widthPct.toFixed(1)}%`;
    barMarker.textContent = `${speedup.toFixed(1)}×`;
    const capped = isSpeedupCapped(n, b);
    status.textContent = speedup >= 3.0
      ? `Memory-bound regime at N = ${n.toLocaleString()}: FlashAttention-2's HBM savings dominate. ${formatBytes(naive)} naive vs ${formatBytes(fa)} FA-2 per head.${capped ? " Capped at 5×: tensor-core throughput is now the bottleneck, not HBM." : ""}`
      : speedup >= 2.0
        ? `Balanced regime at N = ${n.toLocaleString()}: both memory and compute optimizations contribute. ${formatBytes(naive)} vs ${formatBytes(fa)} per head.`
        : `Compute-bound regime at N = ${n.toLocaleString()}: naive attention is already efficient; FA-2 overhead reduces the relative gain.`;
  };

  nInput.addEventListener("input", update);
  bInput.addEventListener("input", update);
  const presets = [
    ["#speedup-short", 512, 64],
    ["#speedup-medium", 2048, 64],
    ["#speedup-long", 8192, 64],
    ["#speedup-ultra", 32768, 64],
  ];
  for (const [selector, n, b] of presets) {
    const btn = document.querySelector(selector);
    if (btn) btn.addEventListener("click", () => { nInput.value = n; bInput.value = b; update(); });
  }
  update();
}

function initializeAttentionComparator() {
  const root = document.querySelector("#attention-compare");
  if (!root) return;
  const buttons = [...document.querySelectorAll("[data-compare-mode]")];
  const results = document.querySelector("#compare-results");
  const status = document.querySelector("#compare-status");
  const pickEl = document.querySelector("#compare-pick");
  const memoryEl = document.querySelector("#compare-memory");
  const exactEl = document.querySelector("#compare-exact");
  if (!results) return;

  const variants = {
    short: [
      { name: "CuDNN attention", kind: "cuDNN", speed: 1.0, memory: "High", exact: "Yes" },
      { name: "FlashAttention-2", kind: "fa2", speed: 1.1, memory: "Medium", exact: "Yes" },
      { name: "Sparse attention", kind: "sparse", speed: 0.9, memory: "Low", exact: "No" },
    ],
    medium: [
      { name: "CuDNN attention", kind: "cuDNN", speed: 0.7, memory: "High", exact: "Yes" },
      { name: "FlashAttention-2", kind: "fa2", speed: 1.0, memory: "Low", exact: "Yes" },
      { name: "FlashDecoding", kind: "decoding", speed: 1.1, memory: "Low", exact: "Yes" },
    ],
    long: [
      { name: "CuDNN attention", kind: "cuDNN", speed: 0.3, memory: "Very high", exact: "Yes" },
      { name: "FlashAttention-2", kind: "fa2", speed: 1.0, memory: "Low", exact: "Yes" },
      { name: "Ring attention", kind: "ring", speed: 0.9, memory: "Very low", exact: "Yes" },
    ],
    extreme: [
      { name: "FlashAttention-2", kind: "fa2", speed: 0.5, memory: "Medium", exact: "Yes" },
      { name: "Context parallel", kind: "cp", speed: 0.8, memory: "Low", exact: "Yes" },
      { name: "Linear attention", kind: "linear", speed: 1.2, memory: "Very low", exact: "No" },
    ],
  };

  const maxSpeed = 1.5;
  const pick = (data) => data.reduce((best, current) => (current.speed > best.speed ? current : best));

  const render = (mode) => {
    const data = variants[mode];
    const recommended = pick(data);
    results.replaceChildren();
    for (const variant of data) {
      const row = document.createElement("div");
      row.className = "compare-row";
      if (variant.name === recommended.name) row.classList.add("recommended");
      row.setAttribute("role", "listitem");

      const name = document.createElement("div");
      name.className = "compare-name";
      const label = document.createElement("span");
      label.textContent = variant.name;
      const kind = document.createElement("small");
      kind.textContent = variant.kind;
      name.append(label, kind);
      if (variant.name === recommended.name) {
        const badge = document.createElement("span");
        badge.className = "compare-badge";
        badge.textContent = "best";
        name.append(badge);
      }

      const bar = document.createElement("div");
      bar.className = "compare-bar";
      const fill = document.createElement("span");
      fill.style.width = `${(Math.min(variant.speed, maxSpeed) / maxSpeed * 100).toFixed(1)}%`;
      bar.append(fill);

      const speed = document.createElement("div");
      speed.className = "compare-speed";
      speed.textContent = `${variant.speed.toFixed(1)}×`;

      row.append(name, bar, speed);
      results.append(row);
    }
    if (pickEl) pickEl.textContent = recommended.name;
    if (memoryEl) memoryEl.textContent = recommended.memory;
    if (exactEl) exactEl.textContent = recommended.exact;
    status.textContent = `${mode.charAt(0).toUpperCase()}${mode.slice(1)} sequences: ${recommended.name} wins at ${recommended.speed.toFixed(1)}×.`;
    buttons.forEach(button => {
      const selected = button.dataset.compareMode === mode;
      button.classList.toggle("selected", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
  };

  buttons.forEach(button => button.addEventListener("click", () => render(button.dataset.compareMode)));
  render("medium");
}

function initializeBackwardVisualization() {
  const steps = [
    { num: "1", op: "Load ∂L/∂O", tag: "HBM read" },
    { num: "2", op: "Recompute S = QKᵀ/√d", tag: "tile on chip" },
    { num: "3", op: "Softmax P = exp(S−m)", tag: "tile recompute" },
    { num: "4", op: "∂L/∂V = Pᵀ · ∂L/∂O", tag: "matmul · write ∂V" },
    { num: "5", op: "∂L/∂P = ∂L/∂O · Vᵀ", tag: "matmul" },
    { num: "6", op: "∂L/∂S = P ⊙ (∂L/∂P − c)", tag: "pointwise" },
    { num: "7", op: "∂L/∂Q = ∂L/∂S · K / √d", tag: "matmul · write ∂Q" },
    { num: "8", op: "∂L/∂K = ∂L/∂Sᵀ · Q / √d", tag: "matmul · write ∂K" },
  ];

  const fields = {
    step: document.querySelector("#backward-step"),
    compute: document.querySelector("#backward-compute"),
    memory: document.querySelector("#backward-memory"),
    gradient: document.querySelector("#backward-gradient"),
    accumulate: document.querySelector("#backward-accumulate"),
    status: document.querySelector("#backward-status"),
  };
  const pipeline = document.querySelector("#backward-flow");
  const playBtn = document.querySelector("#backward-play-btn");
  if (!fields.step || !pipeline || !playBtn) return;

  const stepCells = [];
  for (const step of steps) {
    const cell = document.createElement("div");
    cell.className = "pipeline-step";
    const num = document.createElement("span");
    num.className = "pipeline-num";
    num.textContent = `step ${step.num}`;
    const op = document.createElement("span");
    op.className = "pipeline-op";
    op.textContent = step.op;
    const tag = document.createElement("span");
    tag.className = "pipeline-tag";
    tag.textContent = step.tag;
    cell.append(num, op, tag);
    stepCells.push(cell);
    pipeline.append(cell);
  }

  const legacy = {
    compute: steps[0].op,
    memory: "Read Q, K, V from HBM",
    gradient: "—",
    accumulate: "—",
  };

  let index = 0;
  let timer;
  const stopPlaying = () => {
    clearInterval(timer);
    timer = undefined;
    playBtn.disabled = false;
    playBtn.setAttribute("aria-pressed", "false");
    playBtn.textContent = "Play";
  };

  const render = () => {
    const state = steps[index - 1];
    if (!state) {
      fields.step.textContent = "—";
      fields.compute.textContent = "—";
      fields.memory.textContent = "—";
      fields.gradient.textContent = "—";
      fields.accumulate.textContent = "—";
      fields.status.textContent = "Start: gradients flow from ∂L/∂O back to Q, K, V through the recomputed softmax.";
      for (const cell of stepCells) cell.classList.remove("active", "done");
      return;
    }
    fields.step.textContent = `${state.num} / ${steps.length}`;
    fields.compute.textContent = state.op;
    fields.memory.textContent = state.tag;
    fields.gradient.textContent = state.num === "4" || state.num === "7" || state.num === "8"
      ? "writes ∂L/∂Q or ∂L/∂K or ∂L/∂V"
      : state.num === "5" ? "∂L/∂P" : state.num === "6" ? "∂L/∂S" : "—";
    fields.accumulate.textContent = state.num === "4" || state.num === "7" || state.num === "8"
      ? "Write to HBM"
      : "on-chip accumulation";
    fields.status.textContent = index === steps.length
      ? "Complete: all gradients computed and written to HBM. Total: 3 matrix multiplies + softmax recomputation."
      : `Step ${state.num}: ${state.op} (${state.tag}).`;
    for (const [i, cell] of stepCells.entries()) {
      cell.classList.toggle("done", i < index);
      cell.classList.toggle("active", i === index - 1);
    }
  };

  const step = () => { index = Math.min(index + 1, steps.length); render(); };
  document.querySelector("#backward-step-btn").addEventListener("click", step);
  document.querySelector("#backward-reset-btn").addEventListener("click", () => { stopPlaying(); index = 0; render(); });
  playBtn.addEventListener("click", () => {
    stopPlaying();
    index = 0;
    step();
    playBtn.disabled = true;
    playBtn.setAttribute("aria-pressed", "true");
    playBtn.textContent = "Playing…";
    timer = setInterval(() => {
      step();
      if (index === steps.length) stopPlaying();
    }, 1000);
  });
  pipeline.closest(".widget").addEventListener("keydown", (event) => {
    if (event.target.tagName === "INPUT" || event.target.tagName === "TEXTAREA") return;
    if (event.key === " " || event.key === "ArrowRight") { event.preventDefault(); step(); }
    else if (event.key === "ArrowLeft") { event.preventDefault(); document.querySelector("#backward-reset-btn").click(); }
    else if (event.key === "r" || event.key === "R") { event.preventDefault(); document.querySelector("#backward-reset-btn").click(); }
  });
  render();
}

if (typeof document !== "undefined") {
  initializeHeroSweep();
  initializeAttentionCalculator();
  initializeStreamingState();
  initializeTileObservatory();
  initializeWorkMap();
  initializeProgressRail();
  initializeSpeedupCalculator();
  initializeAttentionComparator();
  initializeBackwardVisualization();
}
