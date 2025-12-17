from __future__ import annotations

import argparse
import os
import time

import numpy as np
import tkinter as tk
import torch

from .model import TransformerACConfig, TransformerActorCritic
from .pong_env import PongConfig, PongEnv
from .wrappers import HistoryObsWrapper


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="runs/pong_transformer_200k.pt")
    p.add_argument("--poll-sec", type=float, default=2.0, help="How often to reload latest checkpoint")
    p.add_argument("--scale", type=int, default=1, help="Render scaling factor")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


class Watcher:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cpu")

        self.ckpt_mtime = 0.0
        self.model: TransformerActorCritic | None = None
        self.model_cfg: TransformerACConfig | None = None

        self.env = PongEnv(PongConfig(terminate_on_point=True), render_mode="rgb_array")
        # seq_len is loaded from checkpoint; initialize after first load
        self.env_wrapped: HistoryObsWrapper | None = None
        self.obs = None

        self.rng = np.random.default_rng(args.seed)
        self.last_reload = 0.0

        self.root = tk.Tk()
        self.root.title("Pong RL Watch (auto-reload checkpoint)")

        # Placeholder size; will resize on first frame
        self.canvas = tk.Canvas(self.root, width=800 * args.scale, height=450 * args.scale, highlightthickness=0)
        self.canvas.pack()

        self._photo = None
        self._img_id = None

    def _load_if_updated(self) -> None:
        now = time.time()
        if now - self.last_reload < self.args.poll_sec:
            return
        self.last_reload = now

        if not os.path.exists(self.args.ckpt):
            return

        mtime = os.path.getmtime(self.args.ckpt)
        if mtime <= self.ckpt_mtime:
            return

        ckpt = torch.load(self.args.ckpt, map_location="cpu")
        cfg = TransformerACConfig(**ckpt["model_cfg"])
        model = TransformerActorCritic(cfg).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        self.model = model
        self.model_cfg = cfg
        self.ckpt_mtime = mtime

        if self.env_wrapped is None or self.env_wrapped.seq_len != cfg.seq_len:
            self.env_wrapped = HistoryObsWrapper(self.env, seq_len=cfg.seq_len)
            self.obs, _ = self.env_wrapped.reset(seed=int(self.rng.integers(0, 2**31 - 1)))

        print(f"Reloaded checkpoint: {self.args.ckpt} (seq_len={cfg.seq_len}, params={model.n_params():,})")

    def _policy_action(self, obs: np.ndarray) -> int:
        assert self.model is not None
        with torch.no_grad():
            o = torch.from_numpy(obs[None, ...]).to(self.device)
            logits, _ = self.model(o)
            return int(torch.argmax(logits, dim=-1).item())

    def _draw(self) -> None:
        frame = self.env.render_rgb(scale=self.args.scale)

        # Tk expects a PhotoImage; the most reliable no-deps path is PPM.
        # Convert to PPM bytes.
        h, w, _ = frame.shape
        header = f"P6 {w} {h} 255 ".encode("ascii")
        ppm = header + frame.tobytes()

        self._photo = tk.PhotoImage(data=ppm, format="PPM")
        if self._img_id is None:
            self._img_id = self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        else:
            self.canvas.itemconfig(self._img_id, image=self._photo)

    def tick(self) -> None:
        self._load_if_updated()

        if self.model is None or self.env_wrapped is None or self.obs is None:
            # No checkpoint yet: keep trying.
            self._draw()
            self.root.after(200, self.tick)
            return

        act = self._policy_action(self.obs)
        self.obs, _, term, trunc, _ = self.env_wrapped.step(act)
        if term or trunc:
            self.obs, _ = self.env_wrapped.reset(seed=int(self.rng.integers(0, 2**31 - 1)))

        self._draw()
        delay_ms = max(1, int(1000 / max(1, self.args.fps)))
        self.root.after(delay_ms, self.tick)

    def run(self) -> None:
        self.root.after(1, self.tick)
        self.root.mainloop()


def main() -> None:
    args = parse_args()
    Watcher(args).run()


if __name__ == "__main__":
    main()


