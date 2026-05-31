/**
 * Layer 3 — Robot Boxing web client (Canvas + Socket.IO)
 *
 * Mirrors rl/robot_boxing_env.py combat rules so live play matches training.
 */

// --- Combat constants (synced with robot_boxing_env.py) ---
const RING_MIN = 0;
const RING_MAX = 1;
const STRIKE_DISTANCE = 0.12;
const MOVE_DELTA = 0.04;
const DODGE_DELTA = 0.08;
const MAX_HEALTH = 100;
const MAX_STAMINA = 100;
const JAB_DAMAGE = 12;
const JAB_STAMINA_COST = 15;
const BLOCK_STAMINA_COST = 8;
const DODGE_STAMINA_COST = 12;
const MOVE_STAMINA_COST = 3;
const STAMINA_RECOVERY = 4;

const ACTION_NAMES = ["step_left", "step_right", "jab", "block", "dodge"];
const ACTION_INDEX = Object.fromEntries(ACTION_NAMES.map((n, i) => [n, i]));

const SERVER_URL = window.location.origin.includes("file:")
  ? "http://localhost:8000"
  : window.location.origin;

// --- DOM ---
const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");
const menuOverlay = document.getElementById("menuOverlay");
const endOverlay = document.getElementById("endOverlay");
const endTitle = document.getElementById("endTitle");
const endMessage = document.getElementById("endMessage");
const scoreLine = document.getElementById("scoreLine");
const connectionStatus = document.getElementById("connectionStatus");

const hud = {
  playerHealthBar: document.getElementById("playerHealthBar"),
  playerStaminaBar: document.getElementById("playerStaminaBar"),
  aiHealthBar: document.getElementById("aiHealthBar"),
  aiStaminaBar: document.getElementById("aiStaminaBar"),
  playerHealthText: document.getElementById("playerHealthText"),
  playerStaminaText: document.getElementById("playerStaminaText"),
  aiHealthText: document.getElementById("aiHealthText"),
  aiStaminaText: document.getElementById("aiStaminaText"),
};

// --- Socket ---
const socket = io(SERVER_URL, { transports: ["websocket", "polling"] });

socket.on("connect", () => {
  connectionStatus.textContent = "AI Connected";
  connectionStatus.className = "connected";
});

socket.on("disconnect", () => {
  connectionStatus.textContent = "Disconnected";
  connectionStatus.className = "disconnected";
});

socket.on("connect_error", () => {
  connectionStatus.textContent = "Server Offline";
  connectionStatus.className = "disconnected";
});

// --- Game state ---
const game = {
  running: false,
  playerX: 0.25,
  aiX: 0.75,
  playerHealth: MAX_HEALTH,
  aiHealth: MAX_HEALTH,
  playerStamina: MAX_STAMINA,
  aiStamina: MAX_STAMINA,
  playerAction: "step_left",
  aiAction: "step_right",
  playerHits: 0,
  aiHits: 0,
  frameCount: 0,
  tickInterval: 10,
  animTimer: 0,
  flashText: "",
  flashTimer: 0,
  waitingAi: false,
  aiTimeoutId: null,
};

const keys = {
  left: false,
  right: false,
  jab: false,
  block: false,
};

// --- Input ---
window.addEventListener("keydown", (e) => {
  if (["ArrowLeft", "ArrowRight", "ArrowDown", "Space"].includes(e.code)) {
    e.preventDefault();
  }
  if (e.code === "ArrowLeft" || e.code === "KeyA") keys.left = true;
  if (e.code === "ArrowRight" || e.code === "KeyD") keys.right = true;
  if (e.code === "Space") keys.jab = true;
  if (e.code === "ArrowDown" || e.code === "KeyS") keys.block = true;
});

window.addEventListener("keyup", (e) => {
  if (e.code === "ArrowLeft" || e.code === "KeyA") keys.left = false;
  if (e.code === "ArrowRight" || e.code === "KeyD") keys.right = false;
  if (e.code === "Space") keys.jab = false;
  if (e.code === "ArrowDown" || e.code === "KeyS") keys.block = false;
});

document.getElementById("startBtn").addEventListener("click", startMatch);
document.getElementById("restartBtn").addEventListener("click", startMatch);

function getPlayerActionFromInput() {
  if (keys.jab) return "jab";
  if (keys.block) return "block";
  if (keys.left) return "step_left";
  if (keys.right) return "step_right";
  return game.playerAction;
}

function resetGame() {
  game.playerX = 0.25;
  game.aiX = 0.75;
  game.playerHealth = MAX_HEALTH;
  game.aiHealth = MAX_HEALTH;
  game.playerStamina = MAX_STAMINA;
  game.aiStamina = MAX_STAMINA;
  game.playerAction = "step_left";
  game.aiAction = "step_right";
  game.playerHits = 0;
  game.aiHits = 0;
  game.frameCount = 0;
  game.animTimer = 0;
  game.flashText = "";
  game.flashTimer = 0;
  game.waitingAi = false;
  if (game.aiTimeoutId) {
    clearTimeout(game.aiTimeoutId);
    game.aiTimeoutId = null;
  }
}

function startMatch() {
  resetGame();
  menuOverlay.classList.add("hidden");
  endOverlay.classList.add("hidden");
  game.running = true;
  showFlash("FIGHT!");
}

function endMatch(playerWon) {
  game.running = false;
  endOverlay.classList.remove("hidden");
  endOverlay.classList.remove("victory", "defeat");
  endOverlay.classList.add(playerWon ? "victory" : "defeat");
  endTitle.textContent = playerWon ? "Victory!" : "Defeat";
  endMessage.textContent = playerWon
    ? "You knocked out the robot boxer."
    : "The DQN agent won this round.";
  scoreLine.textContent = `Score — You: ${game.playerHits}  |  AI: ${game.aiHits}`;
}

// --- Combat (mirrors Python env) ---
function inStrikeRange(attackerX, defenderX) {
  return Math.abs(attackerX - defenderX) <= STRIKE_DISTANCE;
}

function applyMovement(isAi, action) {
  let x = isAi ? game.aiX : game.playerX;
  let stamina = isAi ? game.aiStamina : game.playerStamina;

  if (action === "step_left" && stamina >= MOVE_STAMINA_COST) {
    stamina -= MOVE_STAMINA_COST;
    x = Math.max(RING_MIN, x - MOVE_DELTA);
  } else if (action === "step_right" && stamina >= MOVE_STAMINA_COST) {
    stamina -= MOVE_STAMINA_COST;
    x = Math.min(RING_MAX, x + MOVE_DELTA);
  } else if (action === "dodge" && stamina >= DODGE_STAMINA_COST) {
    stamina -= DODGE_STAMINA_COST;
    x = isAi ? Math.min(RING_MAX, x + DODGE_DELTA) : Math.max(RING_MIN, x - DODGE_DELTA);
  } else if (action === "block" && stamina >= BLOCK_STAMINA_COST) {
    stamina -= BLOCK_STAMINA_COST;
  }

  if (isAi) {
    game.aiX = x;
    game.aiStamina = stamina;
  } else {
    game.playerX = x;
    game.playerStamina = stamina;
  }
}

function recoverStamina() {
  game.playerStamina = Math.min(MAX_STAMINA, game.playerStamina + STAMINA_RECOVERY);
  game.aiStamina = Math.min(MAX_STAMINA, game.aiStamina + STAMINA_RECOVERY);
}

function resolveTurn(playerAction, aiAction) {
  let playerJabbed = false;
  let aiJabbed = false;

  game.playerAction = playerAction;
  game.aiAction = aiAction;

  applyMovement(false, playerAction);
  applyMovement(true, aiAction);

  const playerBlocking = playerAction === "block";
  const aiBlocking = aiAction === "block";
  const playerDodging = playerAction === "dodge";
  const aiDodging = aiAction === "dodge";

  // Player jab
  if (playerAction === "jab" && game.playerStamina >= JAB_STAMINA_COST) {
    game.playerStamina -= JAB_STAMINA_COST;
    playerJabbed = true;
    if (inStrikeRange(game.playerX, game.aiX)) {
      if (!aiBlocking && !(aiDodging && Math.random() < 0.6)) {
        game.aiHealth = Math.max(0, game.aiHealth - JAB_DAMAGE);
        game.playerHits += 1;
        showFlash("HIT!");
      } else {
        showFlash("Blocked");
      }
    } else {
      showFlash("Miss");
    }
  }

  // AI jab
  if (aiAction === "jab" && game.aiStamina >= JAB_STAMINA_COST) {
    game.aiStamina -= JAB_STAMINA_COST;
    aiJabbed = true;
    if (inStrikeRange(game.aiX, game.playerX)) {
      if (playerBlocking) {
        showFlash("You Blocked!");
      } else if (playerDodging && Math.random() < 0.6) {
        showFlash("Dodged");
      } else {
        game.playerHealth = Math.max(0, game.playerHealth - JAB_DAMAGE);
        game.aiHits += 1;
        showFlash("AI Hit!");
      }
    }
  }

  if (aiBlocking && playerJabbed && inStrikeRange(game.playerX, game.aiX)) {
    showFlash("AI Blocked");
  }

  recoverStamina();
  updateHud();

  if (game.aiHealth <= 0) endMatch(true);
  else if (game.playerHealth <= 0) endMatch(false);
}

function emitGameTick() {
  if (!game.running || game.waitingAi) return;

  game.playerAction = getPlayerActionFromInput();
  game.waitingAi = true;

  socket.emit("game_tick", {
    player_x: game.playerX,
    ai_x: game.aiX,
    player_health: game.playerHealth,
    ai_health: game.aiHealth,
    player_stamina: game.playerStamina,
    ai_stamina: game.aiStamina,
    player_action: game.playerAction,
  });

  if (game.aiTimeoutId) clearTimeout(game.aiTimeoutId);
  game.aiTimeoutId = setTimeout(() => {
    if (game.waitingAi && game.running) {
      resolveTurn(getPlayerActionFromInput(), "step_left");
      game.waitingAi = false;
    }
  }, 800);
}

socket.on("ai_response", (data) => {
  if (game.aiTimeoutId) {
    clearTimeout(game.aiTimeoutId);
    game.aiTimeoutId = null;
  }
  if (!game.running) {
    game.waitingAi = false;
    return;
  }

  const aiAction = ACTION_NAMES.includes(data.action) ? data.action : "step_left";
  const playerAction = getPlayerActionFromInput();
  resolveTurn(playerAction, aiAction);
  game.waitingAi = false;
});

// --- HUD ---
function updateHud() {
  const ph = (game.playerHealth / MAX_HEALTH) * 100;
  const ps = (game.playerStamina / MAX_STAMINA) * 100;
  const ah = (game.aiHealth / MAX_HEALTH) * 100;
  const as = (game.aiStamina / MAX_STAMINA) * 100;

  hud.playerHealthBar.style.width = `${ph}%`;
  hud.playerStaminaBar.style.width = `${ps}%`;
  hud.aiHealthBar.style.width = `${ah}%`;
  hud.aiStaminaBar.style.width = `${as}%`;
  hud.playerHealthText.textContent = Math.round(game.playerHealth);
  hud.playerStaminaText.textContent = Math.round(game.playerStamina);
  hud.aiHealthText.textContent = Math.round(game.aiHealth);
  hud.aiStaminaText.textContent = Math.round(game.aiStamina);
}

function showFlash(text) {
  game.flashText = text;
  game.flashTimer = 28;
}

// --- Rendering ---
function ringToCanvasX(normX) {
  const padding = 80;
  const w = canvas.width - padding * 2;
  return padding + normX * w;
}

function drawRing() {
  const w = canvas.width;
  const h = canvas.height;

  // Floor
  const floorGrad = ctx.createLinearGradient(0, h * 0.55, 0, h);
  floorGrad.addColorStop(0, "#3e2723");
  floorGrad.addColorStop(1, "#1a1208");
  ctx.fillStyle = floorGrad;
  ctx.fillRect(0, h * 0.5, w, h * 0.5);

  // Ropes / backdrop
  ctx.fillStyle = "#0d1520";
  ctx.fillRect(0, 0, w, h * 0.52);

  // Canvas
  ctx.strokeStyle = "#5d4037";
  ctx.lineWidth = 6;
  ctx.strokeRect(40, h * 0.52, w - 80, h * 0.38);

  ctx.fillStyle = "#4e342e";
  ctx.fillRect(40, h * 0.52, w - 80, h * 0.38);

  // Ropes
  for (let i = 0; i < 4; i++) {
    const y = h * 0.52 + (h * 0.38 * (i + 1)) / 5;
    ctx.strokeStyle = i % 2 === 0 ? "#c62828" : "#1565c0";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(40, y);
    ctx.lineTo(w - 40, y);
    ctx.stroke();
  }

  // Center line
  ctx.setLineDash([8, 8]);
  ctx.strokeStyle = "rgba(255,255,255,0.15)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(w / 2, h * 0.52);
  ctx.lineTo(w / 2, h * 0.9);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawFighter(x, isAi, action) {
  const cx = ringToCanvasX(x);
  const baseY = canvas.height * 0.78;
  const scale = isAi ? 1 : 1;
  const color = isAi ? "#ff7043" : "#4fc3f7";
  const dark = isAi ? "#bf360c" : "#0277bd";

  ctx.save();
  ctx.translate(cx, baseY);
  if (!isAi) ctx.scale(-1, 1);

  // Shadow
  ctx.fillStyle = "rgba(0,0,0,0.35)";
  ctx.beginPath();
  ctx.ellipse(0, 8, 36 * scale, 10, 0, 0, Math.PI * 2);
  ctx.fill();

  // Legs
  ctx.fillStyle = dark;
  ctx.fillRect(-14, -50, 12, 50);
  ctx.fillRect(4, -50, 12, 50);

  // Torso
  ctx.fillStyle = color;
  if (isAi) {
    // Robot body
    ctx.fillRect(-22, -110, 44, 62);
    ctx.fillStyle = "#37474f";
    ctx.fillRect(-18, -100, 36, 20);
    ctx.fillStyle = "#00e5ff";
    ctx.fillRect(-10, -95, 20, 10);
  } else {
    ctx.fillRect(-20, -108, 40, 58);
  }

  // Head
  ctx.fillStyle = isAi ? "#78909c" : "#ffcc80";
  ctx.beginPath();
  ctx.arc(0, -125, 18, 0, Math.PI * 2);
  ctx.fill();
  if (isAi) {
    ctx.fillStyle = "#00e5ff";
    ctx.fillRect(-10, -130, 8, 4);
    ctx.fillRect(4, -130, 8, 4);
  }

  // Arms by action
  ctx.strokeStyle = color;
  ctx.lineWidth = 10;
  ctx.lineCap = "round";

  let armOffset = 0;
  if (action === "jab") armOffset = -25;
  if (action === "block") armOffset = 15;

  ctx.beginPath();
  ctx.moveTo(-18, -95);
  ctx.lineTo(-35, -75 + armOffset);
  ctx.stroke();

  ctx.beginPath();
  ctx.moveTo(18, -95);
  ctx.lineTo(40, -80 - armOffset);
  ctx.stroke();

  if (action === "block") {
    ctx.fillStyle = "rgba(255,255,255,0.25)";
    ctx.fillRect(-30, -100, 50, 40);
  }

  ctx.restore();
}

function drawFlash() {
  if (game.flashTimer <= 0) return;
  game.flashTimer -= 1;
  ctx.save();
  ctx.font = 'bold 42px "Orbitron", sans-serif';
  ctx.textAlign = "center";
  ctx.fillStyle = `rgba(255, 235, 59, ${game.flashTimer / 28})`;
  ctx.fillText(game.flashText, canvas.width / 2, canvas.height * 0.35);
  ctx.restore();
}

function drawAiActionLabel() {
  if (!game.running) return;
  ctx.font = '600 14px "Rajdhani", sans-serif';
  ctx.fillStyle = "#ff7043";
  ctx.textAlign = "right";
  ctx.fillText(`AI: ${game.aiAction.replace("_", " ")}`, canvas.width - 24, 28);
}

function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawRing();
  drawFighter(game.playerX, false, game.playerAction);
  drawFighter(game.aiX, true, game.aiAction);
  drawFlash();
  drawAiActionLabel();
}

function gameLoop() {
  game.frameCount += 1;
  game.animTimer += 1;

  if (game.running) {
    if (game.frameCount % game.tickInterval === 0) {
      emitGameTick();
    }
  }

  render();
  requestAnimationFrame(gameLoop);
}

updateHud();
requestAnimationFrame(gameLoop);
