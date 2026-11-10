"""Evaluate all trained agents on unseen test seasons (2021-2024) and draw the report figures.

    python -m agrisense.evaluate

Outputs in results/:
    eval_results.csv        one row per crop x city x season x year x policy
    summary_by_policy.csv   the headline table
    fig_*.png               charts used in the report, slides and app
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CITIES, CROPS, DEFAULT_SOIL, RESULTS_DIR, STAGE_NAMES
from .env import IrrigationEnv
from .agents import QLearningAgent, StateEncoder, ThresholdPolicy
from .simulate import evaluate, model_path, run_season
from . import weather as wx

# ---- chart style (validated categorical palette; colour always paired with a legend/label)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
POLICY_COLORS = {
    "Q-learning agent (RL)": "#2a78d6",
    "Sensor threshold rule": "#eb6834",
    "Fixed schedule": "#1baf7a",
    "Rainfed (no irrigation)": "#eda100",
}
CITY_COLORS = dict(zip(CITIES, ["#2a78d6", "#eb6834", "#1baf7a"]))
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2,
})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / name, dpi=160)
    plt.close(fig)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("policy").agg(
        avg_irrigation_mm=("irrigation_mm", "mean"),
        avg_irrigation_events=("irrigation_events", "mean"),
        avg_wasted_mm=("wasted_mm", "mean"),
        avg_stress_days=("stress_days", "mean"),
        avg_relative_yield_pct=("relative_yield", lambda s: 100 * s.mean()),
        _y=("yield_t_ha", "sum"), _w=("irrigation_mm", "sum"))
    g["water_productivity_kg_per_m3"] = np.where(g._w > 0, g._y * 1000 / (g._w * 10), np.nan)
    order = list(POLICY_COLORS)
    return g.drop(columns=["_y", "_w"]).reindex(order).round(2)


# ------------------------------------------------------------------------- figures
def fig_climate(weather):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    months = np.arange(1, 13)
    for city, w in weather.items():
        m = w.assign(month=w.date.dt.month, year=w.date.dt.year)
        rain = m.groupby(["year", "month"]).rain_mm.sum().groupby("month").mean()
        et0 = m.groupby(["year", "month"]).et0_mm.sum().groupby("month").mean()
        axes[0].plot(months, rain.values, color=CITY_COLORS[city], label=city, marker="o", ms=4)
        axes[1].plot(months, et0.values, color=CITY_COLORS[city], label=city, marker="o", ms=4)
    for ax, title in zip(axes, ["Average monthly rainfall (mm)", "Average monthly reference ET0 (mm)"]):
        ax.set_title(title, loc="left")
        ax.set_xticks(months, list("JFMAMJJASOND"))
    axes[0].legend()
    _save(fig, "fig_climate.png")


def fig_training(crops):
    fig, axes = plt.subplots(1, len(crops), figsize=(4 * len(crops), 3.4), sharey=False)
    for ax, crop in zip(np.atleast_1d(axes), crops):
        h = pd.read_csv(RESULTS_DIR / f"training_{crop.lower()}.csv")
        ax.plot(h.episode, h.reward.rolling(5).mean(), color=POLICY_COLORS["Q-learning agent (RL)"])
        ax.set_title(f"{crop}: learning curve", loc="left")
        ax.set_xlabel("training episode (1 episode = 1 season)")
        ax.set_ylabel("season reward (mean of 500 seasons)")
    _save(fig, "fig_training_curves.png")


def fig_policy_comparison(summary):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    names = summary.index.tolist()
    colors = [POLICY_COLORS[n] for n in names]
    short = [n.replace(" (no irrigation)", "").replace(" (RL)", "") for n in names]
    for ax, col, title, fmt in [
        (axes[0], "avg_irrigation_mm", "Irrigation water used per season (mm)", "{:.0f}"),
        (axes[1], "avg_relative_yield_pct", "Yield achieved (% of maximum)", "{:.1f}%")]:
        vals = summary[col].values
        bars = ax.barh(short, vals, color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, " " + fmt.format(v),
                    va="center", color=INK, fontsize=9)
        ax.set_title(title, loc="left")
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, vals.max() * 1.18)
    axes[1].set_yticklabels([])
    _save(fig, "fig_policy_comparison.png")


def fig_water_by_city(df):
    pols = ["Q-learning agent (RL)", "Sensor threshold rule", "Fixed schedule"]
    crops = list(CROPS)
    fig, axes = plt.subplots(1, len(crops), figsize=(12, 3.6), sharey=True)
    for ax, crop in zip(axes, crops):
        d = df[df.crop == crop].groupby(["city", "policy"]).irrigation_mm.mean().unstack()
        x = np.arange(len(d))
        for i, p in enumerate(pols):
            ax.bar(x + (i - 1) * 0.27, d[p].values, width=0.27, color=POLICY_COLORS[p],
                   label=p, edgecolor=SURFACE, linewidth=2)
        ax.set_xticks(x, d.index)
        ax.set_title(f"{crop}: water per season (mm)", loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper left", fontsize=8)
    _save(fig, "fig_water_by_city.png")


def fig_season_trace(weather, crop_name="Maize", city="Jodhpur", season="Kharif (monsoon)", year=2023):
    crop = CROPS[crop_name]
    date = f"{year}-{crop.seasons[season]}"
    logs = {}
    for policy in [QLearningAgent.load(model_path(crop_name)), ThresholdPolicy()]:
        env = IrrigationEnv(weather[city], crop, DEFAULT_SOIL, seed=123 + year)
        summ, log = run_season(env, policy, env.index_of(date))
        logs[policy.name] = (summ, log)
    taw, raw = env.taw, env.raw

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.2, 1, 1]))
    ax = axes[0]
    for name, (summ, log) in logs.items():
        ax.plot(log.day, 100 * (1 - log.depletion_mm / taw), color=POLICY_COLORS[name],
                label=f"{name}  ({summ['irrigation_mm']:.0f} mm, yield {100 * summ['relative_yield']:.1f}%)")
    ax.axhline(100 * (1 - raw / taw), color=INK2, ls="--", lw=1)
    ax.text(1, 100 * (1 - raw / taw) + 2, "stress line (RAW used up)", color=INK2, fontsize=8)
    ax.set_ylabel("soil water available (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"{crop_name}, {city}, {season} {year}: one season, day by day", loc="left")
    ax.legend(loc="lower left", fontsize=8)

    first_log = next(iter(logs.values()))[1]
    axes[1].bar(first_log.day, first_log.rain_mm, color=BLUES[3], width=0.8)
    axes[1].set_ylabel("rain (mm)")
    for i, (name, (_, log)) in enumerate(logs.items()):
        irr = log[log.irrigation_mm > 0]
        axes[2].bar(irr.day + (i - 0.5) * 0.8, irr.irrigation_mm, color=POLICY_COLORS[name], width=0.8,
                    label=name)
    axes[2].set_ylabel("irrigation (mm)")
    axes[2].set_xlabel("days after sowing")
    axes[2].legend(fontsize=8, loc="upper left")
    _save(fig, "fig_season_trace.png")


def fig_policy_heatmap(crop_name="Maize"):
    agent = QLearningAgent.load(model_path(crop_name))
    enc = StateEncoder()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    cmap = matplotlib.colors.ListedColormap(["#f2f1ed"] + BLUES[1:])
    norm = matplotlib.colors.BoundaryNorm([-5, 5, 15, 25, 35, 45], cmap.N)
    for ax, stage in zip(axes, [1, 2, 3]):
        grid = agent.policy_grid(stage=stage, et0_level=2)
        ax.imshow(np.nan_to_num(grid, nan=-1), cmap=cmap, norm=norm, aspect="auto", origin="lower")
        for (i, j), v in np.ndenumerate(grid):
            ax.text(j, i, "rare" if np.isnan(v) else f"{v:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if (not np.isnan(v) and v >= 30) else INK)
        ax.set_xticks(range(3), enc.RAIN_LABELS)
        ax.set_xlabel("rain forecast (next 3 days)")
        ax.set_title(f"{STAGE_NAMES[stage]} stage", loc="left")
        ax.grid(False)
    axes[0].set_yticks(range(10), [f"{10 * i}-{10 * i + 10}%" for i in range(10)])
    axes[0].set_ylabel("soil dryness (depletion of TAW)")
    fig.suptitle(f"What the trained agent does ({crop_name}, hot day): irrigation in mm "
                 "('rare' = seen < 50 times in training)",
                 x=0.01, ha="left", fontweight="bold")
    _save(fig, "fig_policy_heatmap.png")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    weather = wx.load_all()
    df = evaluate(weather)
    df.to_csv(RESULTS_DIR / "eval_results.csv", index=False)
    summary = summarise(df)
    summary.to_csv(RESULTS_DIR / "summary_by_policy.csv")
    by_crop = pd.concat({c: summarise(df[df.crop == c]) for c in CROPS})
    by_crop.to_csv(RESULTS_DIR / "summary_by_crop.csv")

    fig_climate(weather)
    fig_training(list(CROPS))
    fig_policy_comparison(summary)
    fig_water_by_city(df)
    fig_season_trace(weather)
    fig_policy_heatmap()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(f"{len(df)} test simulations\n")
    print(summary)
    print("\nBy crop:\n", by_crop)


if __name__ == "__main__":
    main()
