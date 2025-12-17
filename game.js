(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const canvas = $("game");
  const ctx = canvas.getContext("2d", { alpha: false });

  const scoreLeftEl = $("scoreLeft");
  const scoreRightEl = $("scoreRight");
  const overlayEl = $("overlay");
  const hintEl = $("hint");

  const BASE_W = canvas.width;
  const BASE_H = canvas.height;

  const SETTINGS = {
    winScore: 11,
    paddle: {
      w: 12,
      h: 90,
      speed: 420, // px/sec
      inset: 22,
    },
    ball: {
      r: 6,
      serveSpeed: 360,
      maxSpeed: 820,
      speedUpPerHit: 18,
    },
    ai: {
      maxSpeed: 360,
      reaction: 0.22, // 0..1 (higher = more responsive)
      deadZone: 10,
    },
  };

  const Keys = {
    up: false,
    down: false,
  };

  const State = {
    running: false,
    paused: false,
    gameOver: false,
    lastPointBy: 0, // -1 left, +1 right
    scoreL: 0,
    scoreR: 0,
    left: { x: 0, y: 0, vy: 0 },
    right: { x: 0, y: 0, vy: 0, targetY: 0 },
    ball: { x: 0, y: 0, vx: 0, vy: 0, speed: 0 },
    lastT: 0,
    acc: 0,
  };

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function randSign() {
    return Math.random() < 0.5 ? -1 : 1;
  }

  function formatOverlay(lines) {
    return lines.map((l) => `<div>${l}</div>`).join("");
  }

  function setOverlay(html, show = true) {
    overlayEl.innerHTML = html;
    overlayEl.classList.toggle("hidden", !show);
  }

  function setHint(text) {
    hintEl.textContent = text;
  }

  function updateScoreUI() {
    scoreLeftEl.textContent = String(State.scoreL);
    scoreRightEl.textContent = String(State.scoreR);
  }

  function resetPositions() {
    State.left.x = SETTINGS.paddle.inset;
    State.left.y = (BASE_H - SETTINGS.paddle.h) / 2;
    State.left.vy = 0;

    State.right.x = BASE_W - SETTINGS.paddle.inset - SETTINGS.paddle.w;
    State.right.y = (BASE_H - SETTINGS.paddle.h) / 2;
    State.right.vy = 0;
    State.right.targetY = State.right.y;

    State.ball.x = BASE_W / 2;
    State.ball.y = BASE_H / 2;
    State.ball.vx = 0;
    State.ball.vy = 0;
    State.ball.speed = 0;
  }

  function resetMatch() {
    State.scoreL = 0;
    State.scoreR = 0;
    State.gameOver = false;
    State.paused = false;
    State.running = false;
    State.lastPointBy = 0;
    resetPositions();
    updateScoreUI();
    setHint("Space: serve • W/S or ↑/↓: move • P: pause • R: restart");
    setOverlay(
      formatOverlay([
        "PRESS SPACE TO SERVE",
        "<span style='opacity:.75;font-weight:600'>W/S or ↑/↓ to move • P pause • R restart</span>",
      ]),
      true,
    );
  }

  function serve() {
    State.running = true;
    State.paused = false;
    setOverlay("", false);

    const dir = State.lastPointBy === 0 ? randSign() : -State.lastPointBy;
    const angle = (Math.random() * 0.55 - 0.275) * Math.PI; // -~50..50 degrees

    State.ball.speed = SETTINGS.ball.serveSpeed;
    State.ball.vx = Math.cos(angle) * State.ball.speed * dir;
    State.ball.vy = Math.sin(angle) * State.ball.speed;
  }

  function finishPoint(winner) {
    // winner: -1 left, +1 right
    State.lastPointBy = winner;
    State.running = false;

    if (winner === -1) State.scoreL += 1;
    else State.scoreR += 1;

    updateScoreUI();

    if (State.scoreL >= SETTINGS.winScore || State.scoreR >= SETTINGS.winScore) {
      State.gameOver = true;
      const won = State.scoreL > State.scoreR ? "LEFT WINS" : "RIGHT WINS";
      setOverlay(
        formatOverlay([
          won,
          `<span style='opacity:.75;font-weight:600'>Press R to restart</span>`,
        ]),
        true,
      );
      setHint("R: restart");
      beep("win");
      return;
    }

    resetPositions();
    setOverlay(
      formatOverlay([
        winner === -1 ? "LEFT SCORES" : "RIGHT SCORES",
        "<span style='opacity:.75;font-weight:600'>Press SPACE to serve</span>",
      ]),
      true,
    );
    beep("score");
  }

  function togglePause() {
    if (State.gameOver) return;
    if (!State.running) return;
    State.paused = !State.paused;
    if (State.paused) {
      setOverlay(
        formatOverlay([
          "PAUSED",
          "<span style='opacity:.75;font-weight:600'>Press P to resume</span>",
        ]),
        true,
      );
      setHint("P: resume • R: restart");
    } else {
      setOverlay("", false);
      setHint("Space: serve • W/S or ↑/↓: move • P: pause • R: restart");
    }
  }

  function collideWithPaddle(paddle, side) {
    const p = SETTINGS.paddle;
    const b = SETTINGS.ball;
    const ball = State.ball;

    const px = paddle.x;
    const py = paddle.y;
    const pw = p.w;
    const ph = p.h;

    const bx = ball.x;
    const by = ball.y;

    const hitX = bx + b.r > px && bx - b.r < px + pw;
    const hitY = by + b.r > py && by - b.r < py + ph;
    if (!hitX || !hitY) return;

    // Only bounce if ball is moving towards the paddle
    if (side === -1 && ball.vx >= 0) return;
    if (side === +1 && ball.vx <= 0) return;

    const center = py + ph / 2;
    const offset = clamp((by - center) / (ph / 2), -1, 1);
    const maxBounce = 0.75 * Math.PI;
    const angle = offset * (maxBounce / 2);

    ball.speed = clamp(
      ball.speed + SETTINGS.ball.speedUpPerHit,
      SETTINGS.ball.serveSpeed,
      SETTINGS.ball.maxSpeed,
    );

    const away = side === -1 ? +1 : -1;

    // Nudge ball outside paddle to avoid sticking
    if (side === -1) ball.x = px + pw + b.r;
    else ball.x = px - b.r;

    ball.vx = Math.cos(angle) * ball.speed * away;
    ball.vy = Math.sin(angle) * ball.speed;

    // Add a tiny bit of vertical "english" from paddle motion (player only)
    if (side === -1) ball.vy += paddle.vy * 0.12;

    beep("paddle");
  }

  function step(dt) {
    const p = SETTINGS.paddle;
    const b = SETTINGS.ball;

    // Player input (left paddle)
    const up = Keys.up ? 1 : 0;
    const down = Keys.down ? 1 : 0;
    const intent = down - up;
    State.left.vy = intent * p.speed;
    State.left.y = clamp(State.left.y + State.left.vy * dt, 0, BASE_H - p.h);

    // AI (right paddle): ease target towards ball Y with limited speed
    const targetCenter = State.ball.y - p.h / 2;
    State.right.targetY =
      State.right.targetY + (targetCenter - State.right.targetY) * SETTINGS.ai.reaction;

    const desired = State.right.targetY;
    const delta = desired - State.right.y;
    const dead = SETTINGS.ai.deadZone;
    const aiMove =
      Math.abs(delta) < dead
        ? 0
        : clamp(delta, -SETTINGS.ai.maxSpeed * dt, SETTINGS.ai.maxSpeed * dt);
    State.right.y = clamp(State.right.y + aiMove, 0, BASE_H - p.h);

    if (!State.running || State.paused || State.gameOver) return;

    // Ball motion
    State.ball.x += State.ball.vx * dt;
    State.ball.y += State.ball.vy * dt;

    // Wall bounce
    if (State.ball.y - b.r <= 0) {
      State.ball.y = b.r;
      State.ball.vy = Math.abs(State.ball.vy);
      beep("wall");
    } else if (State.ball.y + b.r >= BASE_H) {
      State.ball.y = BASE_H - b.r;
      State.ball.vy = -Math.abs(State.ball.vy);
      beep("wall");
    }

    // Paddle collisions
    collideWithPaddle(State.left, -1);
    collideWithPaddle(State.right, +1);

    // Scoring
    if (State.ball.x + b.r < 0) finishPoint(+1);
    else if (State.ball.x - b.r > BASE_W) finishPoint(-1);
  }

  function draw() {
    // Background
    ctx.fillStyle = "#06070b";
    ctx.fillRect(0, 0, BASE_W, BASE_H);

    // Midline (dashed)
    ctx.fillStyle = "rgba(232,240,255,0.15)";
    const dashH = 16;
    const gap = 10;
    const x = BASE_W / 2 - 2;
    for (let y = 14; y < BASE_H - 14; y += dashH + gap) {
      ctx.fillRect(x, y, 4, dashH);
    }

    // Paddles
    ctx.fillStyle = "#e8f0ff";
    ctx.fillRect(State.left.x, State.left.y, SETTINGS.paddle.w, SETTINGS.paddle.h);
    ctx.fillRect(State.right.x, State.right.y, SETTINGS.paddle.w, SETTINGS.paddle.h);

    // Ball
    ctx.fillRect(
      Math.round(State.ball.x - SETTINGS.ball.r),
      Math.round(State.ball.y - SETTINGS.ball.r),
      SETTINGS.ball.r * 2,
      SETTINGS.ball.r * 2,
    );

    // Subtle vignette
    const grd = ctx.createRadialGradient(
      BASE_W / 2,
      BASE_H / 2,
      40,
      BASE_W / 2,
      BASE_H / 2,
      BASE_W * 0.78,
    );
    grd.addColorStop(0, "rgba(0,0,0,0)");
    grd.addColorStop(1, "rgba(0,0,0,0.35)");
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, BASE_W, BASE_H);
  }

  // Audio (tiny retro beeps)
  let audio;
  function ensureAudio() {
    if (audio) return audio;
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return null;
    audio = new Ctx();
    return audio;
  }

  function beep(kind) {
    const ac = ensureAudio();
    if (!ac) return;
    const now = ac.currentTime;
    const o = ac.createOscillator();
    const g = ac.createGain();

    const preset =
      kind === "paddle"
        ? { f: 520, d: 0.05 }
        : kind === "wall"
          ? { f: 380, d: 0.04 }
          : kind === "score"
            ? { f: 260, d: 0.09 }
            : kind === "win"
              ? { f: 740, d: 0.12 }
              : { f: 440, d: 0.04 };

    o.type = "square";
    o.frequency.setValueAtTime(preset.f, now);
    g.gain.setValueAtTime(0.0001, now);
    g.gain.exponentialRampToValueAtTime(0.06, now + 0.005);
    g.gain.exponentialRampToValueAtTime(0.0001, now + preset.d);

    o.connect(g).connect(ac.destination);
    o.start(now);
    o.stop(now + preset.d + 0.02);
  }

  function loop(t) {
    if (!State.lastT) State.lastT = t;
    const dt = Math.min(0.05, (t - State.lastT) / 1000);
    State.lastT = t;

    State.acc += dt;
    const FIXED = 1 / 120;
    while (State.acc >= FIXED) {
      step(FIXED);
      State.acc -= FIXED;
    }

    draw();
    requestAnimationFrame(loop);
  }

  function onKeyDown(e) {
    if (e.code === "ArrowUp" || e.code === "KeyW") Keys.up = true;
    if (e.code === "ArrowDown" || e.code === "KeyS") Keys.down = true;

    if (e.code === "Space") {
      e.preventDefault();
      if (State.gameOver) return;
      if (!State.running) serve();
    }

    if (e.code === "KeyP") togglePause();
    if (e.code === "KeyR") resetMatch();

    const ac = ensureAudio();
    if (ac && ac.state === "suspended") ac.resume().catch(() => {});
  }

  function onKeyUp(e) {
    if (e.code === "ArrowUp" || e.code === "KeyW") Keys.up = false;
    if (e.code === "ArrowDown" || e.code === "KeyS") Keys.down = false;
  }

  function onPointerDown() {
    const ac = ensureAudio();
    if (ac && ac.state === "suspended") ac.resume().catch(() => {});
    if (State.gameOver) return;
    if (!State.running) serve();
  }

  function init() {
    ctx.imageSmoothingEnabled = false;
    overlayEl.classList.add("hidden");

    window.addEventListener("keydown", onKeyDown, { passive: false });
    window.addEventListener("keyup", onKeyUp);
    canvas.addEventListener("pointerdown", onPointerDown);

    resetMatch();
    requestAnimationFrame(loop);
  }

  init();
})();

