"""Irrigation policies. Every policy has the same interface:  act(obs) -> irrigation in mm.

Baselines (what farmers / simple controllers do today):
  * RainfedPolicy        - never irrigate
  * FixedSchedulePolicy  - fixed amount on a fixed calendar (traditional practice)
  * ThresholdPolicy      - soil-moisture sensor rule: refill when depletion exceeds RAW (FAO advice)
AI policy:
  * QLearningAgent       - tabular Q-learning over a small, human-readable state space
"""
import json

import numpy as np

from .config import FORECAST_DAYS, IRRIGATION_ACTIONS_MM, STAGE_NAMES


class RainfedPolicy:
    name = "Rainfed (no irrigation)"

    def act(self, obs):
        return 0.0


class FixedSchedulePolicy:
    name = "Fixed schedule"

    def __init__(self, every_days: int = 7, amount_mm: float = 40.0):
        self.every_days, self.amount_mm = every_days, amount_mm

    def act(self, obs):
        return self.amount_mm if obs["day"] % self.every_days == 0 else 0.0


class ThresholdPolicy:
    name = "Sensor threshold rule"

    def act(self, obs):
        # refill the root zone to field capacity once the readily available water is used up
        return obs["depletion_mm"] if obs["depletion_mm"] > obs["raw"] else 0.0


# --------------------------------------------------------------------------- Q-learning
class StateEncoder:
    """Turns the continuous observation into one of 10 x 4 x 3 x 3 = 360 discrete states.

    Each feature is something a farmer would also look at:
      soil dryness (10 levels), crop growth stage (4), rain forecast (3), evaporative demand (3).
    """
    RAIN_BINS = np.array([2.0, 15.0])              # none / light / heavy  (mm over forecast window)
    ET0_BINS = np.array([3.0, 5.0])                # low / medium / high  (mm/day)
    SHAPE = (10, 4, 3, 3)

    RAIN_LABELS = ["<2 mm", "2-15 mm", ">15 mm"]
    ET0_LABELS = ["low (<3)", "medium (3-5)", "high (>5)"]

    @property
    def n_states(self):
        return int(np.prod(self.SHAPE))

    def features(self, obs):
        # plain comparisons instead of np.digitize: same result, ~20x faster in the training loop
        d = min(9, int(obs["depletion_frac"] * 10))
        rain = obs["rain_forecast_mm"]
        r = 0 if rain < self.RAIN_BINS[0] else (1 if rain < self.RAIN_BINS[1] else 2)
        et0 = obs["et0_mm"]
        e = 0 if et0 < self.ET0_BINS[0] else (1 if et0 < self.ET0_BINS[1] else 2)
        return d, int(obs["stage"]), r, e

    def encode(self, obs) -> int:
        d, g, r, e = self.features(obs)
        return ((d * 4 + g) * 3 + r) * 3 + e


class QLearningAgent:
    name = "Q-learning agent (RL)"

    def __init__(self, actions=IRRIGATION_ACTIONS_MM, alpha=0.1, gamma=0.95, seed=0):
        self.encoder = StateEncoder()
        self.actions = np.array(actions, dtype=float)
        self.alpha, self.gamma = alpha, gamma
        self.q = np.zeros((self.encoder.n_states, len(self.actions)))
        self.visits = np.zeros_like(self.q, dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        self.epsilon = 0.0   # 0 = always pick the best known action (used after training)

    def choose(self, state: int) -> int:
        """Epsilon-greedy action index."""
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(self.actions)))
        return int(self.q[state].argmax())  # ties -> smallest irrigation (actions sorted ascending)

    def act(self, obs):
        return float(self.actions[self.choose(self.encoder.encode(obs))])

    def update(self, s, a, r, s_next, done):
        """Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]"""
        target = r if done else r + self.gamma * self.q[s_next].max()
        self.q[s, a] += self.alpha * (target - self.q[s, a])
        self.visits[s, a] += 1

    # ---- inspection helper (used by the website and report to show the learned policy)
    def policy_grid(self, stage: int, et0_level: int, min_visits: int = 50) -> np.ndarray:
        """Best irrigation (mm) for every (dryness level, rain-forecast level).

        Situations met fewer than `min_visits` times in training are returned as NaN ("rare"):
        the agent has not had enough experience there for its choice to mean much."""
        q = self.q.reshape(*self.encoder.SHAPE, -1)[:, stage, :, et0_level, :]
        seen = self.visits.reshape(*self.encoder.SHAPE, -1)[:, stage, :, et0_level, :].sum(-1) >= min_visits
        grid = self.actions[q.argmax(-1)]
        return np.where(seen, grid, np.nan)

    # ---- persistence
    def save(self, path, meta: dict | None = None):
        np.savez_compressed(path, q=self.q, visits=self.visits, actions=self.actions,
                            meta=json.dumps(meta or {}))

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        agent = cls(actions=d["actions"])
        agent.q, agent.visits = d["q"], d["visits"]
        agent.meta = json.loads(str(d["meta"]))
        return agent


__all__ = ["RainfedPolicy", "FixedSchedulePolicy", "ThresholdPolicy", "QLearningAgent",
           "StateEncoder", "STAGE_NAMES", "FORECAST_DAYS"]
