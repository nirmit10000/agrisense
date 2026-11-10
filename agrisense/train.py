"""Train one tabular Q-learning agent per crop.

Each training episode = one full crop season in a random city, starting on a random
day of a random training year (2010-2020). Random start dates mean the agent sees
monsoon, winter and summer conditions in all three climates.

    python -m agrisense.train            # all crops
    python -m agrisense.train Maize      # one crop
"""
import sys
import time

import numpy as np
import pandas as pd

from .config import (CROPS, DEFAULT_SOIL, MODEL_DIR, RESULTS_DIR, TRAIN_YEARS,
                     RewardConfig, TrainConfig)
from .env import IrrigationEnv
from .agents import QLearningAgent
from .simulate import model_path
from . import weather as wx


def train_crop(crop_name: str, cfg: TrainConfig = TrainConfig(), reward=RewardConfig(),
               weather_by_city: dict | None = None, verbose=True):
    crop = CROPS[crop_name]
    weather_by_city = weather_by_city or wx.load_all()
    rng = np.random.default_rng(cfg.seed)
    envs = [IrrigationEnv(w, crop, DEFAULT_SOIL, reward=reward, seed=cfg.seed + i)
            for i, w in enumerate(weather_by_city.values())]
    starts = [e.valid_start_indices(TRAIN_YEARS) for e in envs]

    agent = QLearningAgent(alpha=cfg.alpha_start, gamma=cfg.gamma, seed=cfg.seed)
    decay_episodes = int(cfg.episodes * cfg.eps_decay_frac)
    history, t_start = [], time.time()

    for ep in range(cfg.episodes):
        progress = min(1.0, ep / decay_episodes)
        agent.epsilon = cfg.eps_start + (cfg.eps_end - cfg.eps_start) * progress
        agent.alpha = cfg.alpha_start + (cfg.alpha_end - cfg.alpha_start) * progress
        k = rng.integers(len(envs))
        env = envs[k]
        # start the season with a random soil moisture so every dryness level gets explored
        obs = env.reset(rng.choice(starts[k]), initial_depletion_frac=rng.uniform(0, 0.6))
        s = agent.encoder.encode(obs)
        total, water, done = 0.0, 0.0, False
        while not done:
            a = agent.choose(s)
            obs, r, done, info = env.step(agent.actions[a])
            s_next = agent.encoder.encode(obs)
            agent.update(s, a, r, s_next, done)
            s, total, water = s_next, total + r, water + info["irrigation_mm"]
        history.append(dict(episode=ep, reward=total, irrigation_mm=water,
                            relative_yield=env.relative_yield(), epsilon=agent.epsilon))
        if verbose and (ep + 1) % 10000 == 0:
            recent = pd.DataFrame(history[-1000:])
            print(f"  [{crop_name}] ep {ep + 1:>6}  eps={agent.epsilon:.2f}  "
                  f"reward={recent.reward.mean():8.1f}  water={recent.irrigation_mm.mean():6.1f} mm  "
                  f"yield={recent.relative_yield.mean():.3f}  ({time.time() - t_start:.0f}s)")

    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    agent.save(model_path(crop_name), meta=dict(crop=crop_name, soil=DEFAULT_SOIL,
                                                 episodes=cfg.episodes, reward=reward.__dict__))
    # keep the log small: one row = average of 100 consecutive episodes
    log = pd.DataFrame(history)
    log = log.groupby(log.episode // 100).mean().round(3)
    log.to_csv(RESULTS_DIR / f"training_{crop_name.lower()}.csv", index=False)
    return agent


if __name__ == "__main__":
    names = sys.argv[1:] or list(CROPS)
    weather = wx.load_all()
    for name in names:
        print(f"Training {name} ...")
        train_crop(name, weather_by_city=weather)
