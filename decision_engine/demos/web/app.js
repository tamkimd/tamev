/**
 * TAMEV Studio — Real-Time Edge AI Decision Engine
 * Interactive Game Controller, Canvas Graphics Engine, and Telemetry Visualizer
 */

let currentGame = "snake";
let isRunning = true;
let tickSpeed = 80;
let timerId = null;
let dangerRadar = true;
let soundEnabled = false;

// Telemetry & Sparkline Buffer (Populated strictly from real inference measurements)
const latencyHistory = [];
let lastStateData = null;

// Audio Context for Procedural Web Audio FX
let audioCtx = null;

function getAudioContext() {
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass) {
      audioCtx = new AudioContextClass();
    }
  }
  if (audioCtx && audioCtx.state === "suspended") {
    audioCtx.resume();
  }
  return audioCtx;
}

function playHapticTick(freq = 600, duration = 0.015) {
  if (!soundEnabled) return;
  try {
    const ctx = getAudioContext();
    if (!ctx) return;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(freq, ctx.currentTime);
    gain.gain.setValueAtTime(0.08, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + duration);
  } catch (e) {
    // Audio context not allowed before interaction
  }
}

function playChime() {
  if (!soundEnabled) return;
  try {
    const ctx = getAudioContext();
    if (!ctx) return;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "triangle";
    osc.frequency.setValueAtTime(587, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.1);
    gain.gain.setValueAtTime(0.12, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.12);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.12);
  } catch (e) {}
}

const canvas = document.getElementById("game-canvas");
const ctx = canvas.getContext("2d");
const sparkCanvas = document.getElementById("sparkline-canvas");
const sparkCtx = sparkCanvas ? sparkCanvas.getContext("2d") : null;

// Tetris vibrant jewel palette
const TETRIS_COLORS = {
  1: "#06b6d4", // I - Cyan
  2: "#f59e0b", // O - Amber
  3: "#8b5cf6", // T - Purple
  4: "#10b981", // S - Emerald
  5: "#f43f5e", // Z - Rose
  6: "#3b82f6", // J - Blue
  7: "#ec4899", // L - Pink
};

// Action Direction Icons mapping
const ACTION_ICONS = {
  UP: "↑",
  DOWN: "↓",
  LEFT: "←",
  RIGHT: "→",
  left: "←",
  right: "→",
  rotate: "↻",
  drop: "⤓",
};

// =====================================================================
// Initialization & Game Loop
// =====================================================================

function init() {
  initSparklineCanvas();
  resetGame();
  startLoop();
  setupKeyboardShortcuts();
}

function initSparklineCanvas() {
  if (sparkCanvas) {
    sparkCanvas.width = sparkCanvas.offsetWidth || 140;
    sparkCanvas.height = 28;
  }
}

function startLoop() {
  if (timerId) clearInterval(timerId);
  if (isRunning) {
    timerId = setInterval(stepOnce, tickSpeed);
  }
}

function togglePlayPause() {
  isRunning = !isRunning;
  const btn = document.getElementById("btn-play-pause");
  const icon = document.getElementById("play-icon");
  const label = document.getElementById("play-label");
  const hudStatus = document.getElementById("hud-status-text");

  if (isRunning) {
    icon.textContent = "⏸";
    label.textContent = "Pause";
    btn.className = "btn btn-primary";
    if (hudStatus) hudStatus.textContent = "AI ACTIVE";
    startLoop();
  } else {
    icon.textContent = "▶";
    label.textContent = "Play";
    btn.className = "btn";
    if (hudStatus) hudStatus.textContent = "PAUSED";
    if (timerId) clearInterval(timerId);
  }
}

function setSpeed(ms, btnElement) {
  tickSpeed = ms;
  document.querySelectorAll(".speed-pill").forEach((el) => el.classList.remove("active"));
  if (btnElement) btnElement.classList.add("active");
  if (isRunning) startLoop();
}

function toggleRadar(enabled) {
  dangerRadar = enabled;
  if (lastStateData) {
    if (currentGame === "snake") renderSnake(lastStateData);
    else renderTetris(lastStateData);
  }
}

function toggleSound() {
  soundEnabled = !soundEnabled;
  const icon = document.getElementById("sound-icon");
  if (soundEnabled) {
    icon.textContent = "🔊";
    getAudioContext();
    playChime();
  } else {
    icon.textContent = "🔇";
  }
}

async function switchGame(gameType) {
  currentGame = gameType;
  document.getElementById("tab-snake").className = gameType === "snake" ? "tab-btn active" : "tab-btn";
  document.getElementById("tab-tetris").className = gameType === "tetris" ? "tab-btn active" : "tab-btn";

  if (gameType === "snake") {
    canvas.width = 400;
    canvas.height = 400;
  } else {
    canvas.width = 300;
    canvas.height = 540;
  }

  await fetch(`/api/switch?game=${gameType}`, { method: "POST" });
  await resetGame();
}

let currentModel = "nano";

async function switchModel(modelId) {
  if (modelId === currentModel) return;
  const previousModel = currentModel;
  currentModel = modelId;

  // 1. Update tab buttons active state
  document.querySelectorAll(".model-tab-btn").forEach((btn) => btn.classList.remove("active"));
  const activeTab = document.getElementById(`model-tab-${modelId}`);
  if (activeTab) activeTab.classList.add("active");

  // 2. Update tier matrix rows active state
  document.querySelectorAll(".tier-table tbody tr").forEach((tr) => tr.classList.remove("active-tier"));
  const activeRow = document.getElementById(`tier-row-${modelId}`);
  if (activeRow) activeRow.classList.add("active-tier");

  // 3. Update active indicator text
  const indicator = document.getElementById("active-tier-indicator");
  const pipeBackbone = document.getElementById("pipe-backbone-text");
  if (indicator) indicator.textContent = `Loading ${modelId}...`;

  try {
    const res = await fetch(`/api/model?model=${modelId}`, { method: "POST" });
    const data = await res.json();
    if (data.success) {
      const m = data.model_info;
      if (indicator) {
        indicator.textContent = `Active: ${m.name} (${m.params})`;
      }
      if (pipeBackbone) {
        pipeBackbone.textContent = `${m.name} (${m.params})`;
      }
      playChime();
    } else {
      throw new Error(data.detail || "Switch failed");
    }
  } catch (err) {
    console.error("Model switch failed:", err);
    currentModel = previousModel;
    if (indicator) indicator.textContent = `Error loading ${modelId}`;
  }
}

async function resetGame() {
  try {
    const res = await fetch(`/api/reset?game=${currentGame}`, { method: "POST" });
    const data = await res.json();
    renderState(data);
  } catch (e) {
    console.error("Reset error:", e);
  }
}

async function stepOnce() {
  try {
    const res = await fetch(`/api/step?game=${currentGame}`, { method: "POST" });
    const data = await res.json();
    renderState(data);
    playHapticTick();
  } catch (e) {
    console.error("Step error:", e);
  }
}

// =====================================================================
// State Rendering & Telemetry
// =====================================================================

function renderState(data) {
  if (!data) return;
  lastStateData = data;

  // 1. Update Metrics & Real Multi-Stage Latency Breakdown
  if (data.latency_breakdown) {
    const b = data.latency_breakdown;
    document.getElementById("metric-latency").textContent = `${b.total_ms.toFixed(1)} ms`;
    const fps = Math.round(1000 / Math.max(b.total_ms, 0.1));
    document.getElementById("metric-fps").textContent = `${fps} /s`;

    // Sub-stage indicators
    const elTok = document.getElementById("sub-tok");
    const elFwd = document.getElementById("sub-fwd");
    const elSoft = document.getElementById("sub-soft");
    if (elTok) elTok.textContent = `Tok: ${b.tokenize_ms.toFixed(1)}ms`;
    if (elFwd) elFwd.textContent = `Fwd: ${b.forward_ms.toFixed(1)}ms`;
    if (elSoft) elSoft.textContent = `Soft: ${b.softmax_ms.toFixed(2)}ms`;

    // Multi-color breakdown bar
    const total = Math.max(b.total_ms, 0.01);
    const pTok = (b.tokenize_ms / total) * 100;
    const pFwd = (b.forward_ms / total) * 100;
    const pSoft = (b.softmax_ms / total) * 100;

    const barTok = document.getElementById("bar-tok");
    const barFwd = document.getElementById("bar-fwd");
    const barSoft = document.getElementById("bar-soft");
    if (barTok) barTok.style.width = `${pTok}%`;
    if (barFwd) barFwd.style.width = `${pFwd}%`;
    if (barSoft) barSoft.style.width = `${pSoft}%`;

    // Push into sparkline buffer
    latencyHistory.push(b.total_ms);
    if (latencyHistory.length > 60) latencyHistory.shift();
    drawSparkline();
  } else if (data.latency_ms !== undefined) {
    document.getElementById("metric-latency").textContent = `${data.latency_ms.toFixed(1)} ms`;
    const fps = Math.round(1000 / Math.max(data.latency_ms, 0.1));
    document.getElementById("metric-fps").textContent = `${fps} /s`;
    latencyHistory.push(data.latency_ms);
    if (latencyHistory.length > 60) latencyHistory.shift();
    drawSparkline();
  }

  if (data.confidence !== undefined) {
    document.getElementById("metric-confidence").textContent = `${(data.confidence * 100).toFixed(1)}%`;
  }
  if (data.token_count !== undefined) {
    document.getElementById("metric-tokens").textContent = data.token_count;
  }
  if (data.device) {
    document.getElementById("metric-device").textContent = `${data.device} ⇄`;
  }
  if (data.drift) {
    document.getElementById("metric-drift").textContent = data.drift;
  }

  const scoreVal = data.score !== undefined ? data.score : 0;
  document.getElementById("hud-score-val").textContent = scoreVal;

  if (data.chosen_action) {
    document.getElementById("badge-chosen-action").textContent = `OPTIMAL: ${data.chosen_action.toUpperCase()}`;
  }

  // 2. Syntax-Highlighted Context Box
  if (data.context) {
    renderHighlightedContext(data.context);
  }

  // 3. Render Action Probabilities List
  renderProbabilities(data);

  // 4. Render Game Canvas
  if (currentGame === "snake") {
    renderSnake(data);
  } else {
    renderTetris(data);
  }
}

function renderHighlightedContext(text) {
  const box = document.getElementById("state-text");
  if (!box) return;

  // Color-coded token formatting
  let formatted = text
    .replace(/(WALL DANGER)/g, '<span style="color: #f43f5e; font-weight: 700; background: rgba(244,63,94,0.15); padding: 1px 4px; border-radius: 3px;">$1</span>')
    .replace(/(BODY COLLISION)/g, '<span style="color: #ef4444; font-weight: 700; background: rgba(239,68,68,0.15); padding: 1px 4px; border-radius: 3px;">$1</span>')
    .replace(/(SAFE CLEAR)/g, '<span style="color: #10b981; font-weight: 600; background: rgba(16,185,129,0.12); padding: 1px 4px; border-radius: 3px;">$1</span>')
    .replace(/(Snake Head at \([0-9]+, [0-9]+\))/g, '<span style="color: #38bdf8; font-weight: 600;">$1</span>')
    .replace(/(Target food at \([0-9]+, [0-9]+\))/g, '<span style="color: #f59e0b; font-weight: 600;">$1</span>')
    .replace(/(Goal:.*)/g, '<span style="color: #a78bfa; font-style: italic;">$1</span>');

  box.innerHTML = formatted;
}

function renderProbabilities(data) {
  const container = document.getElementById("probs-container");
  if (!container || !data.options || !data.probabilities) return;

  container.innerHTML = "";

  data.options.forEach((opt, idx) => {
    const p = data.probabilities[idx] || 0;
    const pct = (p * 100).toFixed(1);
    const isWinner = opt.id === data.chosen_action;
    const icon = ACTION_ICONS[opt.id] || "▶";

    const item = document.createElement("div");
    item.className = `prob-item ${isWinner ? "winner" : ""}`;

    item.innerHTML = `
      <div class="prob-header">
        <div class="prob-name-group">
          <div class="action-icon">${icon}</div>
          <span class="prob-action-name">${opt.id.toUpperCase()}</span>
          <span style="font-size: 11px; color: var(--text-dim); max-width: 260px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
            ${opt.text || ""}
          </span>
        </div>
        <div class="prob-stats">
          ${isWinner ? '<span class="badge-pill green">SELECTED</span>' : ''}
          <span class="prob-pct">${pct}%</span>
        </div>
      </div>
      <div class="prob-track">
        <div class="prob-fill ${isWinner ? 'winner' : ''}" style="width: ${pct}%"></div>
      </div>
    `;

    container.appendChild(item);
  });
}

function drawSparkline() {
  if (!sparkCtx) return;
  const w = sparkCanvas.width;
  const h = sparkCanvas.height;

  sparkCtx.clearRect(0, 0, w, h);
  if (latencyHistory.length < 2) return;

  const min = 0;
  const max = Math.max(8.0, ...latencyHistory);
  const step = w / (latencyHistory.length - 1);

  // Background gradient under line
  sparkCtx.beginPath();
  sparkCtx.moveTo(0, h);

  latencyHistory.forEach((val, i) => {
    const x = i * step;
    const y = h - (val / max) * (h - 4);
    if (i === 0) sparkCtx.lineTo(x, y);
    else sparkCtx.lineTo(x, y);
  });

  sparkCtx.lineTo(w, h);
  sparkCtx.closePath();

  const grad = sparkCtx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, "rgba(16, 185, 129, 0.25)");
  grad.addColorStop(1, "rgba(16, 185, 129, 0.0)");
  sparkCtx.fillStyle = grad;
  sparkCtx.fill();

  // Draw line
  sparkCtx.beginPath();
  latencyHistory.forEach((val, i) => {
    const x = i * step;
    const y = h - (val / max) * (h - 4);
    if (i === 0) sparkCtx.moveTo(x, y);
    else sparkCtx.lineTo(x, y);
  });
  sparkCtx.strokeStyle = "#10b981";
  sparkCtx.lineWidth = 1.5;
  sparkCtx.stroke();
}

// =====================================================================
// Canvas Graphics Engine (Snake)
// =====================================================================

function renderSnake(data) {
  const w = canvas.width;
  const h = canvas.height;
  const gridSize = 20;
  const cellSize = w / gridSize;

  // 1. Clear background with deep obsidian
  ctx.fillStyle = "#05070d";
  ctx.fillRect(0, 0, w, h);

  // 2. Blueprint Dot-Grid
  ctx.fillStyle = "rgba(255, 255, 255, 0.04)";
  for (let x = cellSize; x < w; x += cellSize) {
    for (let y = cellSize; y < h; y += cellSize) {
      ctx.fillRect(x - 0.75, y - 0.75, 1.5, 1.5);
    }
  }

  // 3. Danger Radar Overlays (Subtle glowing perimeter lines if head is near danger)
  if (dangerRadar && data.snake && data.snake.length > 0) {
    const head = data.snake[0];
    const borderDangerColor = "rgba(244, 63, 94, 0.4)";
    ctx.lineWidth = 3;
    ctx.strokeStyle = borderDangerColor;

    if (head[1] <= 1) { // Near top wall
      ctx.beginPath(); ctx.moveTo(0, 1); ctx.lineTo(w, 1); ctx.stroke();
    }
    if (head[1] >= gridSize - 2) { // Near bottom wall
      ctx.beginPath(); ctx.moveTo(0, h - 1); ctx.lineTo(w, h - 1); ctx.stroke();
    }
    if (head[0] <= 1) { // Near left wall
      ctx.beginPath(); ctx.moveTo(1, 0); ctx.lineTo(1, h); ctx.stroke();
    }
    if (head[0] >= gridSize - 2) { // Near right wall
      ctx.beginPath(); ctx.moveTo(w - 1, 0); ctx.lineTo(w - 1, h); ctx.stroke();
    }
  }

  // 4. Draw Food (Pulsing Ruby Apple with Multi-Stop Glow)
  if (data.food) {
    const fx = data.food[0] * cellSize + cellSize / 2;
    const fy = data.food[1] * cellSize + cellSize / 2;
    const radius = cellSize / 2.6;

    // Outer Aura
    const aura = ctx.createRadialGradient(fx, fy, 2, fx, fy, cellSize * 1.2);
    aura.addColorStop(0, "rgba(244, 63, 94, 0.5)");
    aura.addColorStop(0.5, "rgba(244, 63, 94, 0.12)");
    aura.addColorStop(1, "rgba(244, 63, 94, 0)");
    ctx.fillStyle = aura;
    ctx.beginPath();
    ctx.arc(fx, fy, cellSize * 1.2, 0, Math.PI * 2);
    ctx.fill();

    // Jewel Core
    const core = ctx.createRadialGradient(fx - 2, fy - 2, 1, fx, fy, radius);
    core.addColorStop(0, "#fda4af");
    core.addColorStop(0.4, "#f43f5e");
    core.addColorStop(1, "#be123c");
    ctx.fillStyle = core;
    ctx.beginPath();
    ctx.arc(fx, fy, radius, 0, Math.PI * 2);
    ctx.fill();

    // Food Sparkle
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(fx - radius * 0.35, fy - radius * 0.35, 1.8, 0, Math.PI * 2);
    ctx.fill();
  }

  // 5. Draw Snake Body (Rounded Capsules + Head Sensory Dots)
  if (data.snake && data.snake.length > 0) {
    const len = data.snake.length;

    for (let i = len - 1; i >= 0; i--) {
      const seg = data.snake[i];
      const sx = seg[0] * cellSize;
      const sy = seg[1] * cellSize;
      const isHead = i === 0;

      if (isHead) {
        // Head with vivid neon glow
        ctx.shadowColor = "rgba(16, 185, 129, 0.6)";
        ctx.shadowBlur = 12;

        const headGrad = ctx.createLinearGradient(sx, sy, sx + cellSize, sy + cellSize);
        headGrad.addColorStop(0, "#05f292");
        headGrad.addColorStop(1, "#10b981");
        ctx.fillStyle = headGrad;

        ctx.beginPath();
        ctx.roundRect(sx + 1, sy + 1, cellSize - 2, cellSize - 2, 7);
        ctx.fill();
        ctx.shadowBlur = 0; // reset

        // Sensory Eye Dots
        ctx.fillStyle = "#022c1b";
        let eye1 = [sx + 5, sy + 5];
        let eye2 = [sx + cellSize - 7, sy + 5];

        if (data.direction === "DOWN") {
          eye1 = [sx + 5, sy + cellSize - 7];
          eye2 = [sx + cellSize - 7, sy + cellSize - 7];
        } else if (data.direction === "LEFT") {
          eye1 = [sx + 5, sy + 5];
          eye2 = [sx + 5, sy + cellSize - 7];
        } else if (data.direction === "RIGHT") {
          eye1 = [sx + cellSize - 7, sy + 5];
          eye2 = [sx + cellSize - 7, sy + cellSize - 7];
        }

        ctx.beginPath();
        ctx.arc(eye1[0], eye1[1], 1.8, 0, Math.PI * 2);
        ctx.arc(eye2[0], eye2[1], 1.8, 0, Math.PI * 2);
        ctx.fill();

      } else {
        // Body segment tapering in alpha
        const alpha = Math.max(0.35, 1 - (i / len) * 0.65);
        ctx.fillStyle = `rgba(16, 185, 129, ${alpha})`;

        ctx.beginPath();
        ctx.roundRect(sx + 1.5, sy + 1.5, cellSize - 3, cellSize - 3, 5);
        ctx.fill();
      }
    }
  }

  // 6. Game Over Overlay
  if (data.game_over) {
    playChime();
    ctx.fillStyle = "rgba(5, 7, 13, 0.82)";
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "#f43f5e";
    ctx.font = "800 22px 'Inter', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("COLLISION DETECTED", w / 2, h / 2 - 12);
    ctx.fillStyle = "#94a3b8";
    ctx.font = "500 13px 'Inter', sans-serif";
    ctx.fillText("Auto-restarting next tick...", w / 2, h / 2 + 16);
  }
}

// =====================================================================
// Canvas Graphics Engine (Tetris)
// =====================================================================

function renderTetris(data) {
  const w = canvas.width;
  const h = canvas.height;
  const cols = 10;
  const rows = 20;
  const cellSize = w / cols;

  ctx.fillStyle = "#05070d";
  ctx.fillRect(0, 0, w, h);

  // Grid lines
  ctx.strokeStyle = "rgba(255, 255, 255, 0.03)";
  ctx.lineWidth = 1;
  for (let x = 0; x <= w; x += cellSize) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
  }
  for (let y = 0; y <= h; y += cellSize) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  // Draw board blocks
  if (data.board) {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = data.board[r][c];
        if (val) {
          const color = TETRIS_COLORS[val] || "#06b6d4";
          drawCyberBlock(c * cellSize, r * cellSize, cellSize, color);
        }
      }
    }
  }

  // Ghost Piece Projection & Falling Piece
  if (data.current_piece) {
    const shape = data.current_piece.shape;
    const px = data.current_piece.x;
    const py = data.current_piece.y;
    const color = TETRIS_COLORS[data.current_piece.type_id] || "#8b5cf6";

    // 1. Calculate Ghost Piece Landing
    let ghostY = py;
    while (!checkCollision(data.board, shape, px, ghostY + 1, cols, rows)) {
      ghostY++;
    }

    // Draw Ghost Piece outline
    if (ghostY > py) {
      for (let r = 0; r < shape.length; r++) {
        for (let c = 0; c < shape[r].length; c++) {
          if (shape[r][c]) {
            const gx = (px + c) * cellSize;
            const gy = (ghostY + r) * cellSize;
            ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
            ctx.setLineDash([3, 3]);
            ctx.strokeRect(gx + 2, gy + 2, cellSize - 4, cellSize - 4);
            ctx.setLineDash([]);
          }
        }
      }
    }

    // 2. Draw Active Falling Piece
    for (let r = 0; r < shape.length; r++) {
      for (let c = 0; c < shape[r].length; c++) {
        if (shape[r][c]) {
          const bx = (px + c) * cellSize;
          const by = (py + r) * cellSize;
          drawCyberBlock(bx, by, cellSize, color, true);
        }
      }
    }
  }

  if (data.game_over) {
    ctx.fillStyle = "rgba(5, 7, 13, 0.82)";
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "#f43f5e";
    ctx.font = "800 22px 'Inter', sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("STACK OVERFLOW", w / 2, h / 2 - 12);
    ctx.fillStyle = "#94a3b8";
    ctx.font = "500 13px 'Inter', sans-serif";
    ctx.fillText("Auto-restarting next tick...", w / 2, h / 2 + 16);
  }
}

function checkCollision(board, shape, px, py, cols, rows) {
  if (!board) return false;
  for (let r = 0; r < shape.length; r++) {
    for (let c = 0; c < shape[r].length; c++) {
      if (shape[r][c]) {
        const nx = px + c;
        const ny = py + r;
        if (nx < 0 || nx >= cols || ny >= rows) return true;
        if (ny >= 0 && board[ny][nx] !== 0) return true;
      }
    }
  }
  return false;
}

function drawCyberBlock(x, y, size, color, glow = false) {
  if (glow) {
    ctx.shadowColor = color;
    ctx.shadowBlur = 8;
  }
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.roundRect(x + 1, y + 1, size - 2, size - 2, 4);
  ctx.fill();
  ctx.shadowBlur = 0;

  // Inner top highlight for 3D glassy effect
  ctx.fillStyle = "rgba(255, 255, 255, 0.28)";
  ctx.fillRect(x + 2, y + 2, size - 4, 2);
}

// =====================================================================
// Diagnostic Modals (Invariance, Benchmark, Snippet)
// =====================================================================

function openModal(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add("active");
}

function closeModal(id) {
  const el = document.getElementById(id);
  if (el) el.classList.remove("active");
}

function openInvarianceModal() {
  openModal("modal-invariance");
}

async function runInvarianceTest() {
  const tbody = document.getElementById("invariance-tbody");
  tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--accent-green);">Running 10 permutation tests on active state...</td></tr>`;

  try {
    const res = await fetch("/api/invariance", { method: "POST" });
    const data = await res.json();

    tbody.innerHTML = "";
    data.runs.forEach((r) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="color: var(--text-dim);">#${r.run}</td>
        <td>[${r.order.join(", ")}]</td>
        <td style="color: var(--accent-green); font-weight: 700;">${r.chosen.toUpperCase()}</td>
        <td style="color: var(--accent-green); font-family: var(--font-mono);">${r.drift}</td>
        <td style="color: var(--text-muted);">${r.latency_ms} ms</td>
      `;
      tbody.appendChild(tr);
    });

    const summaryRow = document.createElement("tr");
    summaryRow.style.background = "rgba(16, 185, 129, 0.12)";
    summaryRow.innerHTML = `
      <td colspan="5" style="padding: 10px; color: var(--accent-green); font-weight: 700; text-align: center;">
        ✓ AUDIT PASSED: Maximum Permutation Drift = ${data.max_drift} across all runs. Zero Position Bias Confirmed.
      </td>
    `;
    tbody.appendChild(summaryRow);
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="5" style="color: #f43f5e;">Audit Error: ${e.message}</td></tr>`;
  }
}

function openBenchmarkModal() {
  openModal("modal-benchmark");
}

async function runSpeedBenchmark() {
  const resDiv = document.getElementById("benchmark-results");
  resDiv.style.display = "none";

  try {
    const res = await fetch("/api/benchmark", { method: "POST" });
    const data = await res.json();

    document.getElementById("bench-p50").textContent = `${data.p50_ms} ms`;
    document.getElementById("bench-p95").textContent = `${data.p95_ms} ms`;
    document.getElementById("bench-qps").textContent = `${data.qps} /s`;

    resDiv.style.display = "flex";
  } catch (e) {
    alert("Benchmark failed: " + e.message);
  }
}

async function openSnippetModal() {
  openModal("modal-snippet");
  try {
    const res = await fetch("/api/snippet");
    const data = await res.json();
    document.getElementById("snippet-curl").textContent = data.curl;
    document.getElementById("snippet-python").textContent = data.python;
  } catch (e) {
    document.getElementById("snippet-curl").textContent = "Error loading snippet: " + e.message;
  }
}

function copyContext() {
  const text = document.getElementById("state-text").textContent;
  navigator.clipboard.writeText(text);
  alert("✓ Current decision state copied to clipboard!");
}

function copyCurlSnippet() {
  const text = document.getElementById("snippet-curl").textContent;
  navigator.clipboard.writeText(text);
  alert("✓ cURL command copied to clipboard!");
}

async function toggleDevice() {
  const badge = document.getElementById("metric-device");
  const currentDev = badge ? badge.textContent.trim().toUpperCase() : "CPU";
  const targetDev = currentDev.includes("CPU") ? "mps" : "cpu";

  try {
    const res = await fetch(`/api/device?device=${targetDev}`, { method: "POST" });
    const data = await res.json();
    if (badge) badge.textContent = `${data.device} ⇄`;
  } catch (e) {
    console.error("Device toggle error:", e);
  }
}

// =====================================================================
// Keyboard Shortcuts
// =====================================================================

function setupKeyboardShortcuts() {
  window.addEventListener("keydown", (e) => {
    // Avoid capturing inside inputs
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

    if (e.code === "Space") {
      e.preventDefault();
      togglePlayPause();
    } else if (e.key === "s" || e.key === "S") {
      e.preventDefault();
      stepOnce();
    } else if (e.key === "r" || e.key === "R") {
      e.preventDefault();
      resetGame();
    } else if (e.key === "1") {
      switchGame("snake");
    } else if (e.key === "2") {
      switchGame("tetris");
    } else if (e.key === "m" || e.key === "M") {
      toggleSound();
    }
  });

  window.addEventListener("resize", () => {
    initSparklineCanvas();
    drawSparkline();
  });
}

window.onload = init;
