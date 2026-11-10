"""Field simulator: a daily soil-water balance following FAO-56 (single crop coefficient).

The environment follows the usual RL interface:
    obs = env.reset(start_index)
    obs, reward, done, info = env.step(irrigation_mm)

It exposes *continuous* observations (a dict); turning them into a discrete state is
the agent's job. That keeps the simulator reusable for other agents (e.g. a DQN later).

Water balance for one day (all in mm, Dr = root-zone depletion below field capacity):
    Dr_after_input = Dr - rain - irrigation           (water entering the soil)
    drainage       = water that overfills the root zone (Dr_after_input < 0) and is lost
    Ks             = 1 if Dr <= RAW else (TAW - Dr) / (TAW - RAW)   (water-stress factor)
    ETc            = Kc * ET0                         (what an unstressed crop would use)
    ETa            = Ks * ETc                         (what the crop actually uses)
    Dr_next        = Dr_after_input + ETa
Reward = -(water used + 10 x crop water deficit + water wasted + fixed cost per irrigation)
Season yield (FAO-33):  Ya / Ym = 1 - Ky * (1 - sum(ETa) / sum(ETc))
"""
import numpy as np
import pandas as pd

from .config import (DEFAULT_SOIL, FORECAST_DAYS, FORECAST_ERROR, SOILS, Crop,
                     RewardConfig)


class IrrigationEnv:
    def __init__(self, weather: pd.DataFrame, crop: Crop, soil_name: str = DEFAULT_SOIL,
                 reward: RewardConfig = RewardConfig(), forecast_error: float = FORECAST_ERROR,
                 seed: int | None = None):
        self.weather = weather.reset_index(drop=True)
        self.rain = self.weather["rain_mm"].to_numpy(dtype=float)
        self.et0 = self.weather["et0_mm"].to_numpy(dtype=float)
        self.dates = self.weather["date"].to_numpy()
        # rain over today + next days, precomputed for speed
        self.rain_window = np.convolve(self.rain, np.ones(FORECAST_DAYS), "full")[FORECAST_DAYS - 1:]
        self.crop = crop
        self.soil = SOILS[soil_name]
        self.reward_cfg = reward
        self.forecast_error = forecast_error
        self.rng = np.random.default_rng(seed)

        self.taw = self.soil.taw(crop.root_depth_m)   # total available water, mm
        self.raw = crop.p * self.taw                  # readily available water, mm

    # ------------------------------------------------------------------ helpers
    def valid_start_indices(self, years) -> np.ndarray:
        """Row indices where a full season fits inside the data and starts in `years`."""
        last_start = len(self.weather) - self.crop.season_length - FORECAST_DAYS
        idx = np.arange(last_start)
        yrs = self.weather["date"].dt.year.to_numpy()[:last_start]
        return idx[np.isin(yrs, list(years))]

    def index_of(self, date) -> int:
        return int(self.weather.index[self.weather["date"] == pd.Timestamp(date)][0])

    def _forecast(self) -> float:
        """Rain expected over today + next days, with a random forecast error."""
        t = self.t0 + self.day
        true_rain = self.rain_window[t]
        if self.forecast_error > 0 and true_rain > 0:
            true_rain *= max(0.0, 1 + self.forecast_error * self.rng.standard_normal())
        return float(true_rain)

    def _obs(self) -> dict:
        t = self.t0 + self.day
        return dict(
            day=self.day,
            stage=self.crop.stage(self.day),
            depletion_mm=self.dr,
            depletion_frac=self.dr / self.taw,
            taw=self.taw,
            raw=self.raw,
            rain_forecast_mm=self._forecast(),
            et0_mm=self.et0[t],   # today's evaporative demand (from forecast)
        )

    # ------------------------------------------------------------------ RL API
    def reset(self, start_index: int, initial_depletion_frac: float = 0.0) -> dict:
        self.t0 = int(start_index)
        self.day = 0
        self.dr = initial_depletion_frac * self.taw
        self.sum_etc = 0.0
        self.sum_eta = 0.0
        return self._obs()

    def step(self, irrigation_mm: float):
        t = self.t0 + self.day
        rain, et0 = self.rain[t], self.et0[t]

        # 1) water in; anything above field capacity drains away
        dr = self.dr - rain - irrigation_mm
        drainage = max(0.0, -dr)
        dr = max(0.0, dr)
        wasted_irrigation = min(irrigation_mm, drainage)

        # 2) crop water use, reduced if the soil is too dry
        ks = 1.0 if dr <= self.raw else max(0.0, (self.taw - dr) / (self.taw - self.raw))
        etc = self.crop.kc(self.day) * et0
        eta = ks * etc
        self.dr = min(self.taw, dr + eta)
        self.sum_etc += etc
        self.sum_eta += eta

        # 3) reward: save water, avoid stress, avoid waste
        deficit = etc - eta
        r = self.reward_cfg
        reward = -(r.water_cost * irrigation_mm + r.stress_cost * deficit
                   + r.waste_cost * wasted_irrigation
                   + (r.event_cost if irrigation_mm > 0 else 0.0))

        info = dict(day=self.day, date=self.dates[t], rain_mm=rain, et0_mm=et0,
                    irrigation_mm=irrigation_mm, drainage_mm=drainage,
                    wasted_irrigation_mm=wasted_irrigation, ks=ks, etc_mm=etc, eta_mm=eta,
                    deficit_mm=deficit, depletion_mm=self.dr, reward=reward)

        self.day += 1
        done = self.day >= self.crop.season_length
        return self._obs(), reward, done, info

    def relative_yield(self) -> float:
        """Fraction of maximum yield achieved so far (FAO-33)."""
        if self.sum_etc == 0:
            return 1.0
        return float(np.clip(1 - self.crop.ky * (1 - self.sum_eta / self.sum_etc), 0, 1))
