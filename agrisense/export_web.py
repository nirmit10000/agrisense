"""Export everything the static website needs into web/data/irrigation.json.

The site (index.html, hosted on GitHub Pages) cannot run Python, so every season is simulated
here with the trained agents and the baselines, using the exact same code as the report.

    python -m agrisense.export_web
"""
import json

import numpy as np
import pandas as pd

from .config import (CITIES, CROPS, DEFAULT_SOIL, IRRIGATION_ACTIONS_MM, RESULTS_DIR, ROOT,
                     STAGE_NAMES, SOILS, TEST_YEARS, RewardConfig, TrainConfig)
from .env import IrrigationEnv
from .agents import StateEncoder
from .simulate import all_policies, run_season
from . import weather as wx

OUT = ROOT / "web" / "data" / "irrigation.json"
YEARS = range(2010, 2025)


def _r(x, nd=1):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def main():
    weather = wx.load_all()
    soil = SOILS[DEFAULT_SOIL]
    out = dict(
        actions_mm=list(IRRIGATION_ACTIONS_MM), stages=STAGE_NAMES, test_years=list(TEST_YEARS),
        train_years=[2010, 2020], reward=RewardConfig().__dict__,
        train=dict(episodes=TrainConfig().episodes, alpha=[TrainConfig().alpha_start, TrainConfig().alpha_end],
                   epsilon=[TrainConfig().eps_start, TrainConfig().eps_end], gamma=TrainConfig().gamma),
        rain_labels=StateEncoder.RAIN_LABELS, et0_labels=StateEncoder.ET0_LABELS,
        cities={}, crops={}, seasons={}, rulebook={}, training={}, summary={},
    )

    for city, w in weather.items():
        m = w.assign(month=w.date.dt.month, year=w.date.dt.year)
        m = m[m.year < 2025]
        out["cities"][city] = dict(
            climate=CITIES[city]["climate"], lat=CITIES[city]["lat"], lon=CITIES[city]["lon"],
            annual_rain=_r(m.groupby("year").rain_mm.sum().mean(), 0),
            annual_et0=_r(m.groupby("year").et0_mm.sum().mean(), 0),
            monthly_rain=[_r(v) for v in m.groupby(["year", "month"]).rain_mm.sum().groupby("month").mean()],
            monthly_et0=[_r(v) for v in m.groupby(["year", "month"]).et0_mm.sum().groupby("month").mean()],
        )

    for crop_name, crop in CROPS.items():
        taw = soil.taw(crop.root_depth_m)
        out["crops"][crop_name] = dict(
            season_length=crop.season_length, stage_days=list(crop.stage_days), seasons=crop.seasons,
            kc=[crop.kc_ini, crop.kc_mid, crop.kc_end], root_depth_m=crop.root_depth_m, p=crop.p, ky=crop.ky,
            max_yield_t_ha=crop.max_yield_t_ha, taw_mm=_r(taw), raw_mm=_r(crop.p * taw),
            stress_line_pct=_r(100 * (1 - crop.p)))
        policies = all_policies(crop_name)
        agent = policies[0]

        # rulebook: [stage][et0 level] -> 10 x 3 grid of best mm (None = rare)
        out["rulebook"][crop_name] = [[[[None if np.isnan(v) else int(v) for v in row]
                                        for row in agent.policy_grid(st, e)]
                                       for e in range(3)] for st in range(4)]
        tr = pd.read_csv(RESULTS_DIR / f"training_{crop_name.lower()}.csv")
        out["training"][crop_name] = dict(episode=[int(v) for v in tr.episode],
                                          reward=[_r(v, 0) for v in tr.reward],
                                          water=[_r(v, 0) for v in tr.irrigation_mm])

        for city, w in weather.items():
            for season, mmdd in crop.seasons.items():
                for year in YEARS:
                    key = f"{crop_name}|{season}|{city}|{year}"
                    entry = dict(sow=f"{year}-{mmdd}", test=year in TEST_YEARS, policies={})
                    for policy in policies:
                        env = IrrigationEnv(w, crop, DEFAULT_SOIL, seed=123 + year)
                        s, log = run_season(env, policy, env.index_of(f"{year}-{mmdd}"))
                        if "rain" not in entry:
                            entry["rain"] = [_r(v) for v in log.rain_mm]
                            entry["et0"] = [_r(v) for v in log.et0_mm]
                        entry["policies"][policy.name] = dict(
                            soil=[int(round(100 * (1 - d / taw))) for d in log.depletion_mm],
                            irr=[int(round(v)) for v in log.irrigation_mm],
                            stress=[int(k < 1) for k in log.ks],
                            water=_r(s["irrigation_mm"], 0), events=s["irrigation_events"],
                            wasted=_r(s["wasted_mm"]), stress_days=s["stress_days"],
                            yield_pct=_r(100 * s["relative_yield"]), yield_t=_r(s["yield_t_ha"], 2),
                            wp=_r(s["water_productivity"], 2))
                    out["seasons"][key] = entry

    out["policies"] = [p.name for p in all_policies("Maize")]
    bp = pd.read_csv(RESULTS_DIR / "summary_by_policy.csv").round(2)
    out["summary"]["by_policy"] = bp.astype(object).where(pd.notna(bp), None).to_dict("records")
    ev = pd.read_csv(RESULTS_DIR / "eval_results.csv")
    out["summary"]["n_test_seasons"] = len(ev) // 4
    out["summary"]["by_city"] = (ev.groupby(["city", "policy"])
                                 .agg(water=("irrigation_mm", "mean"), yield_pct=("relative_yield", "mean"))
                                 .assign(yield_pct=lambda d: 100 * d.yield_pct).round(1).reset_index()
                                 .to_dict("records"))
    out["summary"]["by_crop"] = (ev.groupby(["crop", "policy"])
                                 .agg(water=("irrigation_mm", "mean"), yield_pct=("relative_yield", "mean"))
                                 .assign(yield_pct=lambda d: 100 * d.yield_pct).round(1).reset_index()
                                 .to_dict("records"))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB, {len(out['seasons'])} seasons)")


if __name__ == "__main__":
    main()
