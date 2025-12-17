from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class PongConfig:
    width: float = 800.0
    height: float = 450.0
    dt: float = 1.0 / 60.0

    paddle_w: float = 12.0
    paddle_h: float = 90.0
    paddle_speed: float = 420.0
    paddle_inset: float = 22.0

    ball_r: float = 6.0
    serve_speed: float = 360.0
    max_speed: float = 820.0
    speed_up_per_hit: float = 18.0

    ai_max_speed: float = 360.0
    ai_reaction: float = 0.22
    ai_dead_zone: float = 10.0

    terminate_on_point: bool = True


class PongEnv(gym.Env[np.ndarray, int]):
    """
    Headless Pong environment (agent controls left paddle, right paddle is scripted AI).

    Observation (state-based, normalized to [-1, 1]):
      [ball_x, ball_y, ball_vx, ball_vy, paddle_y, opp_y]

    Action space:
      0: noop
      1: up
      2: down

    Reward (sparse):
      +1 when agent scores (ball exits right side)
      -1 when agent concedes (ball exits left side)
      0 otherwise
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, cfg: PongConfig | None = None, *, render_mode: str | None = None):
        super().__init__()
        self.cfg = cfg or PongConfig()
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        self._x_l = self.cfg.paddle_inset
        self._x_r = self.cfg.width - self.cfg.paddle_inset - self.cfg.paddle_w

        self._paddle_y = 0.0
        self._paddle_vy = 0.0
        self._opp_y = 0.0
        self._opp_target_y = 0.0

        self._ball_x = 0.0
        self._ball_y = 0.0
        self._ball_vx = 0.0
        self._ball_vy = 0.0
        self._ball_speed = 0.0

        self._ep_return = 0.0
        self._ep_len = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        _ = options

        self._ep_return = 0.0
        self._ep_len = 0

        self._paddle_y = (self.cfg.height - self.cfg.paddle_h) * 0.5
        self._paddle_vy = 0.0
        self._opp_y = (self.cfg.height - self.cfg.paddle_h) * 0.5
        self._opp_target_y = self._opp_y

        self._ball_x = self.cfg.width * 0.5
        self._ball_y = self.cfg.height * 0.5

        dir_sign = -1.0 if self.np_random.random() < 0.5 else 1.0
        angle = (self.np_random.random() * 0.55 - 0.275) * math.pi

        self._ball_speed = self.cfg.serve_speed
        self._ball_vx = math.cos(angle) * self._ball_speed * dir_sign
        self._ball_vy = math.sin(angle) * self._ball_speed

        return self._obs(), {}

    def step(self, action: int):
        cfg = self.cfg
        dt = cfg.dt

        # Player paddle
        if action == 1:
            self._paddle_vy = -cfg.paddle_speed
        elif action == 2:
            self._paddle_vy = cfg.paddle_speed
        else:
            self._paddle_vy = 0.0

        self._paddle_y = float(
            np.clip(self._paddle_y + self._paddle_vy * dt, 0.0, cfg.height - cfg.paddle_h)
        )

        # Opponent AI
        target_center = self._ball_y - cfg.paddle_h * 0.5
        self._opp_target_y = self._opp_target_y + (target_center - self._opp_target_y) * cfg.ai_reaction

        delta = self._opp_target_y - self._opp_y
        if abs(delta) < cfg.ai_dead_zone:
            move = 0.0
        else:
            move = float(np.clip(delta, -cfg.ai_max_speed * dt, cfg.ai_max_speed * dt))

        self._opp_y = float(np.clip(self._opp_y + move, 0.0, cfg.height - cfg.paddle_h))

        # Ball motion
        self._ball_x += self._ball_vx * dt
        self._ball_y += self._ball_vy * dt

        # Wall bounce
        if self._ball_y - cfg.ball_r <= 0.0:
            self._ball_y = cfg.ball_r
            self._ball_vy = abs(self._ball_vy)
        elif self._ball_y + cfg.ball_r >= cfg.height:
            self._ball_y = cfg.height - cfg.ball_r
            self._ball_vy = -abs(self._ball_vy)

        # Paddle collisions
        self._collide_paddle(side=-1)
        self._collide_paddle(side=+1)

        terminated = False
        reward = 0.0

        # Scoring (agent is left paddle)
        if self._ball_x + cfg.ball_r < 0.0:
            terminated = True
            reward = -1.0
        elif self._ball_x - cfg.ball_r > cfg.width:
            terminated = True
            reward = +1.0

        truncated = False

        self._ep_return += reward
        self._ep_len += 1

        info: dict = {}
        if terminated or truncated:
            info["episode_return"] = float(self._ep_return)
            info["episode_length"] = int(self._ep_len)
            info["winner"] = "agent" if reward > 0 else "opponent"

        # Optionally continue match (not used by default)
        if not cfg.terminate_on_point and terminated:
            terminated = False
            reward = 0.0
            self.reset(seed=None)

        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    def _collide_paddle(self, side: int) -> None:
        cfg = self.cfg
        px = self._x_l if side == -1 else self._x_r
        py = self._paddle_y if side == -1 else self._opp_y

        hit_x = (self._ball_x + cfg.ball_r > px) and (self._ball_x - cfg.ball_r < px + cfg.paddle_w)
        hit_y = (self._ball_y + cfg.ball_r > py) and (self._ball_y - cfg.ball_r < py + cfg.paddle_h)
        if not (hit_x and hit_y):
            return

        # Only bounce if moving towards that paddle
        if side == -1 and self._ball_vx >= 0.0:
            return
        if side == +1 and self._ball_vx <= 0.0:
            return

        center = py + cfg.paddle_h * 0.5
        offset = float(np.clip((self._ball_y - center) / (cfg.paddle_h * 0.5), -1.0, 1.0))
        max_bounce = 0.75 * math.pi
        angle = offset * (max_bounce * 0.5)

        self._ball_speed = float(np.clip(self._ball_speed + cfg.speed_up_per_hit, cfg.serve_speed, cfg.max_speed))
        away = +1.0 if side == -1 else -1.0

        # Nudge out of paddle to avoid sticking
        if side == -1:
            self._ball_x = px + cfg.paddle_w + cfg.ball_r
        else:
            self._ball_x = px - cfg.ball_r

        self._ball_vx = math.cos(angle) * self._ball_speed * away
        self._ball_vy = math.sin(angle) * self._ball_speed

        # Tiny "english" from player paddle motion
        if side == -1:
            self._ball_vy += self._paddle_vy * 0.12

    def _obs(self) -> np.ndarray:
        cfg = self.cfg
        # Normalize to [-1, 1]
        bx = (self._ball_x / cfg.width) * 2.0 - 1.0
        by = (self._ball_y / cfg.height) * 2.0 - 1.0
        bvx = float(np.clip(self._ball_vx / cfg.max_speed, -1.0, 1.0))
        bvy = float(np.clip(self._ball_vy / cfg.max_speed, -1.0, 1.0))

        py = (self._paddle_y / (cfg.height - cfg.paddle_h)) * 2.0 - 1.0
        oy = (self._opp_y / (cfg.height - cfg.paddle_h)) * 2.0 - 1.0

        return np.array([bx, by, bvx, bvy, py, oy], dtype=np.float32)

    def render(self):
        if self.render_mode is None:
            raise RuntimeError("render_mode is None. Create env with render_mode='rgb_array'.")
        if self.render_mode != "rgb_array":
            raise NotImplementedError(f"Unsupported render_mode={self.render_mode!r}")
        return self.render_rgb()

    def render_rgb(self, *, scale: int = 1) -> np.ndarray:
        """
        Returns an RGB frame (H, W, 3) uint8 for videos/viewers.
        """
        cfg = self.cfg
        w = int(cfg.width) * scale
        h = int(cfg.height) * scale
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Background
        img[:, :] = (6, 7, 11)

        # Midline dashed
        dash_h = int(16 * scale)
        gap = int(10 * scale)
        mx = int(cfg.width * 0.5 * scale)
        for y in range(int(14 * scale), h - int(14 * scale), dash_h + gap):
            img[y : y + dash_h, mx - 2 * scale : mx + 2 * scale] = (130, 150, 190)

        # Paddles + ball
        def rect(x0: float, y0: float, rw: float, rh: float, color: tuple[int, int, int]):
            x1 = int(round((x0 + rw) * scale))
            y1 = int(round((y0 + rh) * scale))
            x0i = int(round(x0 * scale))
            y0i = int(round(y0 * scale))
            x0i = max(0, min(w, x0i))
            y0i = max(0, min(h, y0i))
            x1 = max(0, min(w, x1))
            y1 = max(0, min(h, y1))
            img[y0i:y1, x0i:x1] = color

        white = (232, 240, 255)
        rect(self._x_l, self._paddle_y, cfg.paddle_w, cfg.paddle_h, white)
        rect(self._x_r, self._opp_y, cfg.paddle_w, cfg.paddle_h, white)
        rect(self._ball_x - cfg.ball_r, self._ball_y - cfg.ball_r, cfg.ball_r * 2, cfg.ball_r * 2, white)

        return img


