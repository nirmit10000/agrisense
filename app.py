"""AgriSense dashboard.   Run:  streamlit run app.py"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from agrisense.config import (CITIES, CROPS, DEFAULT_SOIL, IRRIGATION_ACTIONS_MM, RESULTS_DIR,
                              STAGE_NAMES, TEST_YEARS, RewardConfig)
from agrisense.agents import StateEncoder
from agrisense.env import IrrigationEnv
from agrisense.simulate import all_policies, run_season
from agrisense.evaluate import POLICY_COLORS
from agrisense import weather as wx

st.set_page_config(page_title="AgriSense - RL Smart Irrigation", layout="wide")

COLOR_SCALE = alt.Scale(domain=list(POLICY_COLORS), range=list(POLICY_COLORS.values()))


@st.cache_data
def load_weather(city):
    return wx.load(city)


@st.cache_resource
def load_policies(crop):
    return all_policies(crop)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("AgriSense")
st.sidebar.caption("Reinforcement learning for smart irrigation")
crop_name = st.sidebar.selectbox("Crop", list(CROPS))
crop = CROPS[crop_name]
city = st.sidebar.selectbox("Location", list(CITIES), index=list(CITIES).index("Jodhpur"))
season = st.sidebar.selectbox("Sowing season", list(crop.seasons))
year = st.sidebar.selectbox("Year", list(range(2024, 2009, -1)), index=1,
                            format_func=lambda y: f"{y} (unseen test year)" if y in TEST_YEARS else f"{y} (training year)")
st.sidebar.caption(f"Climate: {CITIES[city]['climate']}")
st.sidebar.caption(f"Soil: {DEFAULT_SOIL}  ·  Season length: {crop.season_length} days")

st.title("AgriSense: an RL agent that decides when and how much to irrigate")
tab_sim, tab_results, tab_agent, tab_how = st.tabs(
    ["Season simulator", "Results on all test seasons", "Inside the agent", "How it works"])

# ------------------------------------------------------------------ simulator
with tab_sim:
    weather = load_weather(city)
    sow_date = f"{year}-{crop.seasons[season]}"
    summaries, logs = [], []
    for policy in load_policies(crop_name):
        env = IrrigationEnv(weather, crop, DEFAULT_SOIL, seed=123 + year)
        s, log = run_season(env, policy, env.index_of(sow_date))
        summaries.append(s)
        logs.append(log.assign(policy=policy.name))
    summ = pd.DataFrame(summaries).set_index("policy")
    log = pd.concat(logs)
    log["soil_water_pct"] = 100 * (1 - log.depletion_mm / env.taw)

    rl, thr, fixed = (summ.loc[n] for n in list(POLICY_COLORS)[:3])
    st.subheader(f"{crop_name} sown {pd.Timestamp(sow_date):%d %b %Y} in {city}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RL water used", f"{rl.irrigation_mm:.0f} mm",
              f"{rl.irrigation_mm - fixed.irrigation_mm:+.0f} mm vs fixed schedule", delta_color="inverse")
    c2.metric("RL yield", f"{100 * rl.relative_yield:.1f} %",
              f"{100 * (rl.relative_yield - thr.relative_yield):+.1f} pts vs sensor rule")
    c3.metric("Sensor rule water used", f"{thr.irrigation_mm:.0f} mm")
    c4.metric("Season rainfall", f"{rl.rain_mm:.0f} mm")

    show = st.multiselect("Policies to plot", list(POLICY_COLORS), default=list(POLICY_COLORS)[:2])
    sel = log[log.policy.isin(show)]
    raw_line = 100 * (1 - env.raw / env.taw)

    day_axis = alt.X("day:Q", title="days after sowing",
                     scale=alt.Scale(domain=[0, crop.season_length], nice=False, clamp=True))
    base = alt.Chart(sel).encode(x=day_axis)
    lines = base.mark_line(strokeWidth=2).encode(
        y=alt.Y("soil_water_pct:Q", title="soil water available (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("policy:N", scale=COLOR_SCALE, legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["policy", "day", alt.Tooltip("date:T", format="%d %b %Y"),
                 alt.Tooltip("soil_water_pct:Q", format=".0f", title="soil water %"),
                 alt.Tooltip("irrigation_mm:Q", title="irrigation mm"),
                 alt.Tooltip("rain_mm:Q", format=".1f", title="rain mm")])
    rule = alt.Chart(pd.DataFrame({"y": [raw_line]})).mark_rule(strokeDash=[4, 4], color="#52514e").encode(y="y:Q")
    soil_chart = (lines + rule).properties(
        height=280, title="Soil moisture (dashed line = crop starts to suffer water stress below this)")

    events = sel[sel.irrigation_mm > 0].copy()
    # nudge each policy's bars sideways so same-day irrigations don't hide each other
    offsets = {name: (i - (len(show) - 1) / 2) * 0.6 for i, name in enumerate(show)}
    events["day"] = events.day + events.policy.map(offsets)
    rain = log[log.policy == list(POLICY_COLORS)[0]]
    irr_chart = alt.Chart(events).mark_bar(size=5).encode(
        x=day_axis,
        y=alt.Y("irrigation_mm:Q", title="irrigation (mm)", scale=alt.Scale(domain=[0, 60])),
        color=alt.Color("policy:N", scale=COLOR_SCALE, legend=None),
        tooltip=["policy", "irrigation_mm"]).properties(height=130, title="Irrigation events")
    rain_chart = alt.Chart(rain).mark_bar(size=3, color="#3987e5").encode(
        x=day_axis,
        y=alt.Y("rain_mm:Q", title="rain (mm)"), tooltip=["day", alt.Tooltip("rain_mm:Q", format=".1f")]
    ).properties(height=110, title="Rainfall")
    # one stacked figure with a shared day axis (separate small charts collapse in Streamlit)
    st.altair_chart(alt.vconcat(soil_chart, irr_chart, rain_chart).resolve_scale(x="shared"),
                    width="stretch")

    st.markdown("**Season summary, all policies**")
    st.dataframe(summ.assign(relative_yield=lambda d: (100 * d.relative_yield).round(1))
                 .rename(columns={"relative_yield": "yield % of max"})
                 [["irrigation_mm", "irrigation_events", "wasted_mm", "stress_days", "yield % of max",
                   "yield_t_ha", "water_productivity"]].round(2), width="stretch")

# ------------------------------------------------------------------ results
with tab_results:
    path = RESULTS_DIR / "summary_by_policy.csv"
    if not path.exists():
        st.warning("Run `python -m agrisense.evaluate` first.")
    else:
        n = len(pd.read_csv(RESULTS_DIR / "eval_results.csv"))
        st.markdown(f"Averages over **{n} simulated seasons**: 3 crops × 3 cities × their sowing seasons "
                    f"× test years {TEST_YEARS.start}-{TEST_YEARS.stop - 1} × 4 policies. "
                    "None of these seasons were used for training.")
        st.dataframe(pd.read_csv(path, index_col=0), width="stretch")
        st.image(str(RESULTS_DIR / "fig_policy_comparison.png"))
        st.image(str(RESULTS_DIR / "fig_water_by_city.png"))
        st.markdown("**Per crop**")
        st.dataframe(pd.read_csv(RESULTS_DIR / "summary_by_crop.csv", index_col=[0, 1]),
                     width="stretch")
        st.image(str(RESULTS_DIR / "fig_training_curves.png"))
        st.image(str(RESULTS_DIR / "fig_climate.png"))

# ------------------------------------------------------------------ agent internals
with tab_agent:
    agent = load_policies(crop_name)[0]
    st.markdown("The agent's whole 'brain' is a table: for each situation it stores how good each "
                "irrigation amount is (its Q-value). Below is the **best action** it learned for every "
                "situation. 'rare' means the situation happened fewer than 50 times in training (too rare to trust).")
    a, b = st.columns(2)
    stage = a.selectbox("Crop growth stage", range(4), index=2, format_func=lambda i: STAGE_NAMES[i])
    et0_level = b.selectbox("Evaporative demand today (ET0, mm/day)", range(3), index=2,
                            format_func=lambda i: StateEncoder.ET0_LABELS[i])
    grid = agent.policy_grid(stage, et0_level)
    cols = [f"rain forecast {l}" for l in StateEncoder.RAIN_LABELS]
    rows = [f"soil {10 * i}-{10 * i + 10}% dry" for i in range(10)]
    labels = pd.DataFrame([["rare" if np.isnan(v) else f"{v:.0f} mm" for v in r] for r in grid],
                          index=rows, columns=cols)
    blues = ["#f7fbff", "#cde2fb", "#9ec5f4", "#5598e7", "#256abf"]   # 0,10,20,30,40 mm

    def cell_style(v):
        if np.isnan(v):
            return "background-color: #f2f1ed; color: #8a8985"
        level = int(v // 10)
        return f"background-color: {blues[level]}; color: {'white' if level >= 3 else '#0b0b0b'}"

    styles = pd.DataFrame([[cell_style(v) for v in r] for r in grid], index=rows, columns=cols)
    st.dataframe(labels.style.apply(lambda _: styles, axis=None), width="stretch", height=390)
    st.caption("Read it like a farmer's rulebook: the drier the soil, the more water, and less "
               "(or none) when rain is forecast.")
    stats = agent.meta
    st.markdown(f"Trained for **{stats.get('episodes', '?'):,} seasons** · states visited: "
                f"**{int((agent.visits.sum(1) > 0).sum())} / {agent.encoder.n_states}** · "
                f"reward weights: `{stats.get('reward')}`")

# ------------------------------------------------------------------ explanation
with tab_how:
    r = RewardConfig()
    st.markdown(f"""
### The problem
Agriculture uses about 80% of India's freshwater. Farmers usually irrigate on a fixed calendar, which
wastes water when it rains and still lets the crop dry out in hot spells.

### The idea
Treat irrigation as a **sequential decision problem** and let a reinforcement learning agent learn,
by trial and error in a simulated field, *when* and *how much* to irrigate.

| RL element | In AgriSense |
|---|---|
| **Environment** | A daily soil-water simulator (FAO-56 water balance) driven by **real weather** (Open-Meteo, 2010-2025) |
| **State** (what the agent sees) | soil dryness (10 levels) × crop stage (4) × 3-day rain forecast (3) × evaporative demand (3) = **360 states** |
| **Actions** | irrigate {", ".join(map(str, IRRIGATION_ACTIONS_MM))} mm today |
| **Reward** (per day) | −({r.water_cost} × water used + {r.stress_cost} × crop water deficit + {r.waste_cost} × water drained away + {r.event_cost} per irrigation event) |
| **Algorithm** | Tabular Q-learning, ε-greedy exploration (1 → 0.05), learning rate α decaying 0.1 → 0.01, γ = 0.95, 50 000 training seasons per crop |

**Yield** at the end of the season follows FAO-33: `Ya/Ym = 1 − Ky (1 − ΣETa/ΣETc)`.

### Compared against
* **Rainfed**: never irrigate
* **Fixed schedule**: 40 mm every 7 days (typical farmer calendar)
* **Sensor threshold rule**: refill the soil once a moisture sensor says the safe limit is used up (FAO advice)

### Honest limitations
Simulated field, not a real one; yield model is the simple seasonal FAO-33 equation; forecast is
the real future rain plus random error; one soil type.
""")
