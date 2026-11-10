"""Run policies through seasons and summarise the results."""
import numpy as np
import pandas as pd

from .config import CITIES, CROPS, DEFAULT_SOIL, MODEL_DIR, TEST_YEARS
from .env import IrrigationEnv
from .agents import FixedSchedulePolicy, QLearningAgent, RainfedPolicy, ThresholdPolicy


def run_season(env: IrrigationEnv, policy, start_index: int):
    """Simulate one full season. Returns (summary dict, daily log DataFrame)."""
    obs = env.reset(start_index)
    rows, total_reward, done = [], 0.0, False
    while not done:
        irrigation = policy.act(obs)
        obs, reward, done, info = env.step(irrigation)
        total_reward += reward
        rows.append(info)
    log = pd.DataFrame(rows)
    rel_yield = env.relative_yield()
    water = log["irrigation_mm"].sum()
    yield_t = rel_yield * env.crop.max_yield_t_ha
    summary = dict(
        policy=policy.name,
        irrigation_mm=water,
        irrigation_events=int((log["irrigation_mm"] > 0).sum()),
        rain_mm=log["rain_mm"].sum(),
        wasted_mm=log["wasted_irrigation_mm"].sum(),
        stress_days=int((log["ks"] < 1).sum()),
        relative_yield=rel_yield,
        yield_t_ha=yield_t,
        # kg of produce per cubic metre of irrigation water (1 mm over 1 ha = 10 m^3)
        water_productivity=(yield_t * 1000) / (water * 10) if water > 0 else np.nan,
        total_reward=total_reward,
    )
    return summary, log


def model_path(crop_name: str):
    return MODEL_DIR / f"q_{crop_name.lower()}.npz"


def all_policies(crop_name: str):
    return [QLearningAgent.load(model_path(crop_name)), ThresholdPolicy(),
            FixedSchedulePolicy(), RainfedPolicy()]


def evaluate(weather_by_city: dict, years=TEST_YEARS, crops=None, seed=123) -> pd.DataFrame:
    """Every policy x city x crop x realistic sowing season x test year."""
    rows = []
    for crop_name in crops or CROPS:
        crop = CROPS[crop_name]
        policies = all_policies(crop_name)
        for city, weather in weather_by_city.items():
            for season, mmdd in crop.seasons.items():
                for year in years:
                    for policy in policies:
                        # same seed per season -> every policy sees identical forecast errors
                        env = IrrigationEnv(weather, crop, DEFAULT_SOIL, seed=seed + year)
                        summary, _ = run_season(env, policy, env.index_of(f"{year}-{mmdd}"))
                        rows.append(dict(crop=crop_name, city=city, season=season,
                                         year=year, **summary))
    return pd.DataFrame(rows)
