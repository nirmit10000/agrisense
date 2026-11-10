"""Builds docs/AgriSense_Report.docx from the latest results.   python docs/build_report.py

Irrigation numbers come from results/*.csv; leaf-disease numbers from results/leaf_metrics.json.
Anything not produced yet is written as "[pending]" so the build never fails.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agrisense.config import CITIES, CROPS, IRRIGATION_ACTIONS_MM, RewardConfig, TrainConfig  # noqa: E402

RES = ROOT / "results"
OUT = ROOT / "docs" / "AgriSense_Report.docx"
ACCENT = RGBColor(0x1C, 0x5C, 0xAB)
INK2 = RGBColor(0x52, 0x51, 0x4E)

summary = pd.read_csv(RES / "summary_by_policy.csv", index_col=0)
by_crop = pd.read_csv(RES / "summary_by_crop.csv", index_col=[0, 1])
evals = pd.read_csv(RES / "eval_results.csv")
RL, THR, FIX, RAIN = summary.index.tolist()
n_test = len(evals) // 4
s_rl, s_thr, s_fix = summary.loc[RL], summary.loc[THR], summary.loc[FIX]
save_fix = 100 * (1 - s_rl.avg_irrigation_mm / s_fix.avg_irrigation_mm)
save_thr = 100 * (1 - s_rl.avg_irrigation_mm / s_thr.avg_irrigation_mm)
by_city = evals.groupby(["city", "policy"]).agg(w=("irrigation_mm", "mean"), y=("relative_yield", "mean"))
rw, tc = RewardConfig(), TrainConfig()

SITE_URL = "https://nirmit10000.github.io/agrisense/"
PENDING = "[pending]"
_leaf_path = RES / "leaf_metrics.json"
try:
    leaf = json.loads(_leaf_path.read_text(encoding="utf-8")) if _leaf_path.exists() else None
except (OSError, ValueError):
    leaf = None


def lv(get, fmt="{}"):
    """A leaf-metrics value formatted for the text, or [pending] if it is not available yet."""
    try:
        v = get(leaf)
        return PENDING if v is None else fmt.format(v)
    except (TypeError, KeyError, IndexError, ValueError):
        return PENDING


leaf_n_classes = lv(lambda m: len(m["classes"]))
if leaf_n_classes == PENDING:
    leaf_n_classes = "14"  # planned: maize + tomato classes of PlantVillage
leaf_acc = lv(lambda m: 100 * m["test_accuracy"], "{:.1f}%")
leaf_f1 = lv(lambda m: m["macro_f1"], "{:.3f}")
leaf_n_total = lv(lambda m: sum(m["n_images"].values()), "{:,}")
leaf_n_train = lv(lambda m: m["n_images"]["train"], "{:,}")
leaf_n_val = lv(lambda m: m["n_images"]["val"], "{:,}")
leaf_n_test = lv(lambda m: m["n_images"]["test"], "{:,}")
leaf_input = lv(lambda m: m["input_size"])
leaf_grid = lv(lambda m: int(m["input_size"]) // 32)  # MobileNetV2 downsamples by 32
leaf_ep_head = lv(lambda m: m["epochs"]["head"])
leaf_ep_ft = lv(lambda m: m["epochs"]["finetune"])
leaf_minutes = lv(lambda m: m["train_minutes"], "{:.0f}")
leaf_size = lv(lambda m: m["model_size_mb"], "{:.1f} MB")
leaf_model_file = lv(lambda m: m["model_file"])
leaf_crops = lv(lambda m: " and ".join(sorted({c["crop"] for c in m["classes"]})).lower())
if leaf_crops == PENDING:
    leaf_crops = "maize and tomato"

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
for side in ("left_margin", "right_margin"):
    setattr(sec, side, Cm(2.3))
sec.top_margin = sec.bottom_margin = Cm(2.2)

st = doc.styles
st["Normal"].font.name = "Calibri"
st["Normal"].font.size = Pt(11)
st["Normal"].paragraph_format.space_after = Pt(6)
st["Normal"].paragraph_format.line_spacing = 1.15
for lvl, size in [(1, 15), (2, 12.5)]:
    h = st[f"Heading {lvl}"]
    h.font.name, h.font.size, h.font.color.rgb, h.font.bold = "Calibri", Pt(size), ACCENT, True
    h.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
    h.paragraph_format.space_after = Pt(6)


# ------------------------------------------------------------------ helpers
def para(text="", bold_lead=None, style=None, align=None, italic=False, size=None, color=None):
    p = doc.add_paragraph(style=style)
    if bold_lead:
        p.add_run(bold_lead).bold = True
    r = p.add_run(text)
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    if color:
        r.font.color.rgb = color
    if align:
        p.alignment = align
    return p


def bullets(items):
    for it in items:
        if isinstance(it, tuple):
            para(it[1], bold_lead=it[0], style="List Bullet")
        else:
            para(it, style="List Bullet")


def equation(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.font.name, r.font.size = "Cambria Math", Pt(11)
    r.italic = True
    return p


def shade(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def table(header, rows, widths_cm, bold_first_col=False, highlight_row=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold, r.font.size, r.font.color.rgb = True, Pt(9.5), RGBColor(0xFF, 0xFF, 0xFF)
        shade(c, "1C5CAB")
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(9.5)
            if (bold_first_col and i == 0) or ri == highlight_row:
                r.bold = True
            if ri == highlight_row:
                shade(cells[i], "E3EEFB")
    for ri, row in enumerate(t.rows):
        for i, w in enumerate(widths_cm):
            row.cells[i].width = Cm(w)
            if ri < len(t.rows) - 1:  # keep the (short) table on one page, never orphan the header
                for p in row.cells[i].paragraphs:
                    p.paragraph_format.keep_with_next = True
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def figure(path, caption, width_cm=16):
    if not Path(path).exists():  # figure not produced yet: leave a visible marker, don't crash
        para(f"[pending figure: {Path(path).name}]  {caption}", italic=True, size=9.5, color=INK2,
             align=WD_ALIGN_PARAGRAPH.CENTER)
        return
    doc.add_picture(str(path), width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(caption, italic=True, size=9.5, color=INK2, align=WD_ALIGN_PARAGRAPH.CENTER)


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


fmt_pct = lambda v: f"{v:.1f} %"

# ------------------------------------------------------------------ title page
for _ in range(5):
    doc.add_paragraph()
para("EC-422  ARTIFICIAL INTELLIGENCE  ·  COURSE PROJECT", size=11, color=INK2, align=WD_ALIGN_PARAGRAPH.CENTER)
p = para("AgriSense", size=34, color=ACCENT, align=WD_ALIGN_PARAGRAPH.CENTER)
p.runs[0].bold = True
para("AI for Smart Irrigation and Crop Health Monitoring", size=17, align=WD_ALIGN_PARAGRAPH.CENTER)
para("Reinforcement learning for irrigation scheduling  ·  Deep learning for leaf disease detection",
     size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
para("Domain: Agriculture (intelligent irrigation, crop monitoring and resource optimisation)", size=11.5,
     color=INK2, align=WD_ALIGN_PARAGRAPH.CENTER)
for _ in range(4):
    doc.add_paragraph()
para("Submitted by", size=11, color=INK2, align=WD_ALIGN_PARAGRAPH.CENTER)
para("[Student 1 name]  ·  [Roll number]", size=12.5, align=WD_ALIGN_PARAGRAPH.CENTER)
para("[Student 2 name]  ·  [Roll number]", size=12.5, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
para("Course instructor: [Name]", size=11, color=INK2, align=WD_ALIGN_PARAGRAPH.CENTER)
para("November 2026", size=11, color=INK2, align=WD_ALIGN_PARAGRAPH.CENTER)
page_break()

# ------------------------------------------------------------------ abstract
doc.add_heading("Abstract", 1)
para(f"Irrigation is the largest use of fresh water in India, yet most farmers still water on a fixed "
     f"calendar that ignores rainfall and the crop's actual needs. We frame irrigation scheduling as a "
     f"sequential decision problem and solve it with reinforcement learning (RL). A tabular Q-learning agent "
     f"observes four quantities a farmer would also consider (soil dryness, crop growth stage, a 3-day rain "
     f"forecast and evaporative demand) and decides each day how much water to apply. It learns by "
     f"interacting with a field simulator built on the FAO-56 soil-water balance, driven by 16 years of real "
     f"daily weather for three contrasting Indian climates: Pune (semi-arid), Jodhpur (arid) and Ludhiana "
     f"(sub-tropical). After training on 2010-2020 seasons, the agent was tested on {n_test} unseen seasons "
     f"(2021-2024) for maize, wheat and tomato. Compared with a fixed schedule it used "
     f"{save_fix:.0f}% less irrigation water while raising yield from {s_fix.avg_relative_yield_pct:.1f}% "
     f"to {s_rl.avg_relative_yield_pct:.1f}% of the maximum. Compared with an expert soil-moisture "
     f"threshold rule it used {save_thr:.1f}% less water for a {100 - s_rl.avg_relative_yield_pct:.1f}-point "
     f"yield trade-off, giving the highest water productivity of all policies "
     f"({s_rl.water_productivity_kg_per_m3:.2f} kg of produce per m³). The learned policy is a 360-row table "
     f"that can be inspected directly.")
para(f"The second module monitors crop health. A MobileNetV2 convolutional neural network, adapted by "
     f"transfer learning to {leaf_n_total} PlantVillage leaf images, classifies {leaf_n_classes} {leaf_crops} "
     f"classes (healthy leaves and common diseases) with {leaf_acc} accuracy on held-out test images "
     f"(macro F1 {leaf_f1}). Class activation maps show which part of the leaf each diagnosis is based on. "
     f"Both modules are deployed together as a static website ({SITE_URL}). The irrigation results are "
     f"precomputed, and the disease model runs on the user's own device in the browser, so photos are never "
     f"uploaded.")
para("Keywords: reinforcement learning, Q-learning, irrigation scheduling, FAO-56, convolutional neural "
     "network, transfer learning, MobileNetV2, class activation map, plant disease, precision agriculture",
     italic=True, color=INK2)

# ------------------------------------------------------------------ 1 intro
doc.add_heading("1. Introduction", 1)
para("Agriculture accounts for the overwhelming majority of India's freshwater withdrawals, and groundwater "
     "tables are falling in many farming regions. At the same time, a large share of irrigation water is "
     "wasted: water applied just before a rainfall event, or applied in excess of what the root zone can hold, "
     "drains below the roots and is lost. Under-irrigation during hot spells is just as costly, because "
     "water stress at the wrong time directly reduces yield.")
para("Deciding when and how much to irrigate is therefore a trade-off made every day under uncertainty "
     "(future rain, future heat). Each decision changes the soil moisture available for tomorrow's decision. "
     "This is exactly the kind of problem reinforcement learning is designed for: an agent takes actions in an "
     "environment, receives rewards, and learns a policy that maximises long-term reward rather than "
     "short-term gain.")
para("Water is not the only threat to the harvest. The FAO estimates that up to 40% of global crop production "
     "is lost to plant pests and diseases every year [7]. Many leaf diseases can be recognised from their "
     "visible symptoms, but most smallholder farmers have no plant pathologist nearby, and by the time a "
     "disease is identified it has often spread. A model that recognises the disease from a phone photo "
     "turns that expertise into something anyone can use in the field.")
doc.add_heading("1.1 Problem statement", 2)
para("A. Irrigation: given daily weather, a rain forecast and the soil-moisture state of a field, learn an "
     "irrigation policy that minimises water use and water waste while keeping crop yield close to its "
     "potential, and that works across different crops and climates.")
para("B. Crop health: given a photo of a leaf, identify whether it is healthy or which disease it has, and "
     "show which part of the leaf the decision is based on.")
doc.add_heading("1.2 Objectives", 2)
bullets([
    "Build a realistic yet lightweight field simulator from established agronomy (FAO-56, FAO-33) and real weather data.",
    "Formulate irrigation as a Markov Decision Process and train a Q-learning agent on it.",
    "Compare the agent fairly against three baseline strategies on unseen test years.",
    f"Train a convolutional neural network to recognise healthy and diseased leaves of {leaf_crops}, "
    "and make its decisions explainable with class activation maps.",
    "Make both models explainable, and deploy them together as one website that runs in the browser.",
])
doc.add_heading("1.3 System overview", 2)
table(["Module", "AI technique", "Input", "Output"], [
    ["A. Smart irrigation", "Reinforcement learning (tabular Q-learning)",
     "Soil dryness, crop stage, 3-day rain forecast, evaporative demand", "Irrigation amount for today (0-40 mm)"],
    ["B. Leaf disease detection", "Deep learning (MobileNetV2 CNN, transfer learning)",
     "Photo of a leaf", "Healthy / disease name, confidence, heatmap, advice"],
    ["Website", "Static web app (GitHub Pages)", "User's choices and photos", "Both modules in one interface"],
], [3.3, 4.4, 4.5, 4.2], bold_first_col=True)

# ------------------------------------------------------------------ 2 background
doc.add_heading("2. Background", 1)
doc.add_heading("2.1 Crop water use (FAO-56)", 2)
para("The FAO Irrigation and Drainage Paper 56 [1] is the international standard for estimating crop water "
     "requirements. Its key quantities, all in millimetres of water per day or in the root zone, are:")
bullets([
    ("Reference evapotranspiration (ET0): ", "the water a reference grass surface would lose to the "
     "atmosphere on a given day; it summarises the weather's 'drying power' (temperature, sun, wind, humidity)."),
    ("Crop coefficient (Kc): ", "scales ET0 to a specific crop and growth stage. It is low after sowing, peaks in "
     "mid-season when the canopy is full, and falls as the crop matures. Crop water demand is ETc = Kc × ET0."),
    ("Total available water (TAW): ", "how much water the root zone can store between field capacity (full) "
     "and wilting point (empty); it depends on soil type and root depth."),
    ("Readily available water (RAW = p × TAW): ", "the part of TAW the crop can extract without stress. "
     "Once the soil dries beyond RAW, the crop transpires less than it wants and yield suffers."),
])
doc.add_heading("2.2 Yield response to water (FAO-33)", 2)
para("FAO Paper 33 [2] links water stress to yield through a crop-specific yield response factor Ky:")
equation("1 − Ya / Ym  =  Ky · (1 − ΣETa / ΣETc)")
para("where Ya / Ym is actual over maximum yield, and ΣETa / ΣETc is the fraction of the crop's water demand "
     "that was actually met over the season.")
doc.add_heading("2.3 Reinforcement learning and Q-learning", 2)
para("In RL an agent in state s takes action a, receives reward r and moves to state s′. The goal is a policy "
     "that maximises the expected discounted sum of rewards. Q-learning [3, 4] learns an action-value table "
     "Q(s, a), the expected long-term reward of taking a in s and acting well afterwards, using the update:")
equation("Q(s, a)  ←  Q(s, a) + α · [ r + γ · maxₐ′ Q(s′, a′) − Q(s, a) ]")
para("α is the learning rate and γ the discount factor. Q-learning is model-free (it never needs the "
     "simulator's equations, only experience) and off-policy (it can explore randomly while still learning "
     "the best policy). Exploration uses the ε-greedy rule: with probability ε take a random action, "
     "otherwise take the best known one.")
doc.add_heading("2.4 Convolutional neural networks and transfer learning", 2)
para("A convolutional neural network (CNN) learns image features in layers: early layers respond to edges and "
     "colours, later layers to textures and shapes such as lesions or mould. Training a CNN from scratch needs "
     "millions of labelled images. Transfer learning avoids this: we start from a network already trained on "
     "ImageNet (1.2 million everyday photos), keep its general visual features, and retrain only the later "
     "layers for the new task. This needs far fewer images and minutes rather than days of computing.")
para("MobileNetV2 [10] is a CNN designed for phones. It uses depthwise-separable convolutions and inverted "
     "residual blocks to keep accuracy high with only about 3.5 million parameters, so it is small enough to "
     "run inside a web browser.")

# ------------------------------------------------------------------ 3 data
doc.add_heading("3. Irrigation module: data", 1)
para("Daily weather was obtained from the free Open-Meteo Historical Weather API [5] (ERA5 reanalysis) for "
     "1 January 2010 to 31 December 2025: rainfall, FAO Penman-Monteith reference evapotranspiration ET0, and "
     "maximum and minimum temperature. Three locations were chosen to maximise climatic variety, so the agent "
     "must learn a policy that generalises rather than one tuned to a single place. The whole dataset is about "
     "0.5 MB.")
rows = []
for c in CITIES:
    w = pd.read_csv(ROOT / "data" / f"weather_{c.lower()}.csv", parse_dates=["date"])
    w = w[w.date.dt.year < 2025]
    a = w.groupby(w.date.dt.year).rain_mm.sum()
    rows.append([c, CITIES[c]["climate"], f"{a.mean():.0f}", f"{a.min():.0f}-{a.max():.0f}",
                 f"{w.et0_mm.mean() * 365:.0f}"])
table(["City", "Climate", "Mean annual rain (mm)", "Range (mm)", "Annual ET0 (mm)"], rows,
      [2.3, 7.2, 2.4, 2.2, 2.3], bold_first_col=True)
figure(RES / "fig_climate.png", "Figure 1. Monthly rainfall and evaporative demand in the three locations "
       "(2010-2025 average). Jodhpur has the least rain and the highest ET0; Pune has the strongest monsoon; "
       "Ludhiana has some winter rain.")
para("Train/test split. ", bold_lead=None)
doc.paragraphs[-1].runs[0].bold = True
doc.paragraphs[-1].add_run("Seasons starting in 2010-2020 are used for training; seasons starting in "
                           "2021-2024 are held out and never seen by the agent. Splitting by time (rather than at "
                           "random) is the honest choice for weather data, because neighbouring days are correlated.")

# ------------------------------------------------------------------ 4 method
doc.add_heading("4. Irrigation module: methodology", 1)
doc.add_heading("4.1 Field simulator (the RL environment)", 2)
para("The simulator advances one day at a time using the FAO-56 root-zone water balance. Let Dr be the "
     "root-zone depletion (how many mm below field capacity the soil is). Each day:")
bullets([
    "Rain P and irrigation I enter the soil: Dr ← Dr − P − I. If Dr falls below 0, the excess drains below the roots (deep percolation) and is lost.",
    "A water-stress coefficient is computed: Ks = 1 if Dr ≤ RAW, otherwise Ks = (TAW − Dr) / (TAW − RAW).",
    "The crop uses water: ETc = Kc(day) × ET0 and ETa = Ks × ETc; then Dr ← min(TAW, Dr + ETa).",
    "At the end of the season, yield follows the FAO-33 equation above.",
])
para("Crop parameters (stage lengths, Kc values, root depth, depletion fraction p, Ky) are taken from FAO-56 "
     "and FAO-33 tables. The soil is a sandy loam (field capacity 0.22, wilting point 0.10 m³/m³).")
rows = [[c.name, " / ".join(map(str, c.stage_days)), c.season_length,
         f"{c.kc_ini} / {c.kc_mid} / {c.kc_end}", c.root_depth_m, c.p, c.ky, ", ".join(c.seasons)]
        for c in CROPS.values()]
table(["Crop", "Stage days", "Total", "Kc ini / mid / end", "Root (m)", "p", "Ky", "Sowing seasons tested"],
      rows, [1.6, 2.4, 1.2, 2.6, 1.3, 0.9, 1.0, 5.4], bold_first_col=True)

doc.add_heading("4.2 MDP formulation", 2)
table(["Element", "Definition", "Why"], [
    ["State", "Soil dryness Dr/TAW (10 levels) × growth stage (4) × rain forecast for the next 3 days "
              "(<2, 2-15, >15 mm) × today's ET0 (<3, 3-5, >5 mm) = 360 states",
     "Exactly the information a farmer uses; small enough for a table"],
    ["Action", f"Irrigate {', '.join(map(str, IRRIGATION_ACTIONS_MM))} mm today",
     "Realistic drip / sprinkler doses"],
    ["Reward (daily)", f"−( {rw.water_cost:g}·I + {rw.stress_cost:g}·(ETc − ETa) + {rw.waste_cost:g}·wasted "
                       f"irrigation + {rw.event_cost:g}·[I > 0] )",
     "Water costs money; stress costs more (yield); waste and pump starts are penalised"],
    ["Episode", "One crop season (125-145 days) from a random date, city and initial soil moisture",
     "Covers monsoon, winter and summer in all climates"],
    ["Transition", "FAO-56 simulator with real weather; forecast = true rain ± 25 % random error",
     "Forecasts are useful but imperfect"],
], [2.3, 8.6, 5.5], bold_first_col=True)
para("The stress weight is the key design choice: one millimetre of unmet crop water demand is penalised "
     f"{rw.stress_cost:g}× as much as one millimetre of irrigation. This encodes the farmer's priority "
     "that losing yield is far worse than spending some water. A lower weight would save more water at "
     "the cost of more yield.")

doc.add_heading("4.3 Training", 2)
table(["Hyper-parameter", "Value"], [
    ["Training episodes (seasons)", f"{tc.episodes:,} per crop"],
    ["Learning rate α", f"{tc.alpha_start} → {tc.alpha_end} (linear decay)"],
    ["Discount factor γ", f"{tc.gamma}"],
    ["Exploration ε", f"{tc.eps_start} → {tc.eps_end} over the first {int(tc.eps_decay_frac * 100)} % of episodes"],
    ["Q-table size", "360 states × 5 actions = 1,800 values"],
    ["Training time", "≈ 1.5 minutes per crop on a laptop CPU"],
], [7, 7], bold_first_col=True)
para("One agent is trained per crop, because crops differ in water demand and stress sensitivity. Each episode "
     "picks a random city and start date from the training years, so a single agent must work in all three "
     "climates. Decaying α lets the Q-values settle; with a constant α, actions of almost equal value kept "
     "swapping, which made the learned policy noisy and hard to interpret.")

doc.add_heading("4.4 Baseline policies", 2)
bullets([
    ("Rainfed: ", "never irrigate (lower bound)."),
    ("Fixed schedule: ", "40 mm every 7 days, which mimics a typical farmer calendar."),
    ("Sensor threshold rule: ", "when a soil-moisture sensor shows depletion beyond RAW, refill the root zone "
     "exactly to field capacity. This is the FAO-recommended management rule and a strong expert baseline. "
     "It is given perfect soil-moisture knowledge and any irrigation amount, but it ignores the forecast."),
])
doc.add_heading("4.5 Evaluation protocol", 2)
para(f"Each policy is run on every combination of crop, realistic sowing season, city and test year "
     f"(2021-2024): {n_test} seasons per policy. All policies see identical weather and identical forecast "
     f"errors. Metrics: irrigation water (mm), number of irrigation events, irrigation lost to drainage, days "
     f"with water stress, relative yield, and water productivity (kg of produce per m³ of irrigation water).")

# ------------------------------------------------------------------ 5 results
doc.add_heading("5. Irrigation module: results", 1)
doc.add_heading("5.1 Learning", 2)
figure(RES / "fig_training_curves.png", "Figure 2. Average season reward during training. Reward rises "
       "steadily while exploration decays and then plateaus: the agent has converged.")
doc.add_heading("5.2 Overall comparison on unseen seasons", 2)
rows = []
for name, r in summary.iterrows():
    rows.append([name, f"{r.avg_irrigation_mm:.0f}", f"{r.avg_irrigation_events:.1f}", f"{r.avg_wasted_mm:.1f}",
                 f"{r.avg_stress_days:.1f}", fmt_pct(r.avg_relative_yield_pct),
                 "-" if pd.isna(r.water_productivity_kg_per_m3) else f"{r.water_productivity_kg_per_m3:.2f}"])
table(["Policy", "Water (mm)", "Events", "Wasted (mm)", "Stress days", "Yield", "kg / m³"], rows,
      [4.6, 1.8, 1.5, 2.0, 1.9, 1.9, 1.7], bold_first_col=True, highlight_row=0)
figure(RES / "fig_policy_comparison.png", f"Figure 3. Average over {n_test} unseen test seasons.")
para(f"The RL agent used {s_rl.avg_irrigation_mm:.0f} mm per season against {s_fix.avg_irrigation_mm:.0f} mm "
     f"for the fixed schedule, a {save_fix:.0f}% saving, while raising yield from "
     f"{s_fix.avg_relative_yield_pct:.1f}% to {s_rl.avg_relative_yield_pct:.1f}%. The fixed schedule wastes "
     f"about {s_fix.avg_wasted_mm:.0f} mm per season to drainage, because it waters regardless of rain and soil "
     f"state, and it still causes stress in hot spells. Against the expert threshold rule, the agent used "
     f"{save_thr:.1f}% less water and wasted less ({s_rl.avg_wasted_mm:.1f} vs {s_thr.avg_wasted_mm:.1f} mm), "
     f"at the cost of a {100 - s_rl.avg_relative_yield_pct:.1f}-point lower yield. Its water productivity is the "
     f"best of all policies.")

doc.add_heading("5.3 By crop and by climate", 2)
rows = []
for crop in CROPS:
    for pol in [RL, THR, FIX]:
        r = by_crop.loc[(crop, pol)]
        rows.append([crop if pol == RL else "", pol, f"{r.avg_irrigation_mm:.0f}",
                     fmt_pct(r.avg_relative_yield_pct), f"{r.water_productivity_kg_per_m3:.2f}"])
table(["Crop", "Policy", "Water (mm)", "Yield", "kg / m³"], rows, [2.2, 6, 2.6, 2.4, 2.4], bold_first_col=True)
figure(RES / "fig_water_by_city.png", "Figure 4. Irrigation water per season by crop and city. Water needs differ "
       "by a factor of 3 across climates, and the agent adapts without being told which city it is in.")
rows = []
for c in CITIES:
    rl_, th_, fx_ = (by_city.loc[(c, p)] for p in (RL, THR, FIX))
    rows.append([c, f"{rl_.w:.0f}", fmt_pct(100 * rl_.y), f"{th_.w:.0f}", f"{fx_.w:.0f}",
                 f"{100 * (1 - rl_.w / th_.w):.1f} %"])
table(["City", "RL water", "RL yield", "Threshold water", "Fixed water", "RL saving vs threshold"], rows,
      [2.4, 2.2, 2.2, 3, 2.6, 4], bold_first_col=True)
lud = by_city.loc[("Ludhiana", RL)]
para(f"The agent's behaviour differs by climate. In arid Jodhpur it matches the threshold rule almost exactly "
     f"(little rain to exploit). In Ludhiana it saves the most water but loses the most yield "
     f"({100 * lud.y:.1f}%). A likely cause is the long cold-winter periods of very low ET0, where the coarse "
     f"discrete state loses detail and the agent sometimes waits too long (Section 8).")

doc.add_heading("5.4 One season, day by day", 2)
figure(RES / "fig_season_trace.png", "Figure 5. Maize, Jodhpur, Kharif 2023. Both policies keep the soil above "
       "the stress line; the agent gives smaller, earlier doses and uses less water overall.")

doc.add_heading("5.5 What the agent learned", 2)
figure(RES / "fig_policy_heatmap.png", "Figure 6. The learned policy for maize on a hot day: irrigation (mm) for "
       "each combination of soil dryness and rain forecast.")
para("Because the policy is a table, it can be read like a rulebook. The agent discovered on its own, "
     "without being given any agronomic rule:")
bullets([
    "Do not irrigate while the soil is less than about 60% depleted, which is close to the FAO RAW "
    "threshold (p = 0.55) built into the crop model.",
    "When the soil is 60-80% depleted and no rain is forecast, apply the largest dose (40 mm).",
    "When heavy rain is forecast, irrigate less or not at all. This is where it saves water compared with "
    "the threshold rule, which has no forecast.",
])

# ------------------------------------------------------------------ 6 leaf disease
doc.add_heading("6. Leaf disease detection module", 1)
doc.add_heading("6.1 Motivation", 2)
para("Irrigation keeps the crop supplied with water; disease can still destroy it. Up to 40% of global crop "
     "production is lost to plant pests and diseases every year [7], and early, correct identification is "
     "what allows a farmer to treat a field before a disease spreads. Leaf diseases of maize and tomato (for "
     "example northern leaf blight, common rust, early and late blight) have distinctive visible symptoms, "
     "which makes them a natural task for image classification. We chose the same crops as the irrigation "
     "module so that both modules serve the same farmer.")

doc.add_heading("6.2 Data", 2)
para(f"We used the colour images of the public PlantVillage dataset [8, 9], which contains about 54,000 "
     f"photos of individual leaves, each labelled by plant pathologists. We kept the {leaf_n_classes} classes "
     f"for {leaf_crops} (healthy plus each disease) and capped the number of images per class so that no "
     f"class dominates training and training stays fast on a laptop CPU. The images were split 70 / 15 / 15 "
     f"into training, validation and test sets, stratified so every class keeps the same proportion in each "
     f"set. The test set is used only once, for the final numbers.")
table(["Split", "Images", "Used for"], [
    ["Training (70%)", leaf_n_train, "Learning the weights"],
    ["Validation (15%)", leaf_n_val, "Choosing the epoch / checking over-fitting"],
    ["Test (15%)", leaf_n_test, "Final, unbiased accuracy (touched once)"],
    ["Total", leaf_n_total, f"{leaf_n_classes} classes, {leaf_crops}"],
], [4.0, 3.0, 8.0], bold_first_col=True)
if leaf and leaf.get("classes"):
    rows = [[c.get("crop", ""), c.get("name", ""), "healthy" if c.get("healthy") else "disease"]
            for c in leaf["classes"]]
    table(["Crop", "Class", "Type"], rows, [3.0, 8.5, 3.5], bold_first_col=True)

doc.add_heading("6.3 Model and training", 2)
para(f"The classifier is MobileNetV2 pre-trained on ImageNet, with its final layer replaced by a new "
     f"{leaf_n_classes}-way linear classifier. Images are resized to {leaf_input} × {leaf_input} pixels and "
     f"normalised with the ImageNet statistics the network was pre-trained with. Training has two stages:")
bullets([
    ("Stage 1, linear head: ", f"the pre-trained convolutional layers are frozen and only the new "
     f"classifier is trained ({leaf_ep_head} epochs). The network already 'sees' edges, colours and textures; "
     f"this stage learns which of those features indicate which disease."),
    ("Stage 2, fine-tuning: ", f"the last convolutional blocks are unfrozen and trained together with the "
     f"head at a small learning rate ({leaf_ep_ft} epochs), so the high-level features adapt to leaf "
     f"lesions rather than everyday objects."),
])
para(f"Training took about {leaf_minutes} minutes on a laptop CPU. The final model was exported to the ONNX "
     f"format ({leaf_model_file}, {leaf_size}) so that it can run in a web browser.")
para("Why transfer learning? With a few thousand images, a CNN trained from scratch would over-fit; reusing "
     "ImageNet features gives high accuracy with little data and little computing. Why MobileNetV2? It is "
     "small and fast enough to run on a phone or in a browser, which is where a farmer would use it. Its "
     "architecture (global average pooling followed by one linear layer) also makes class activation maps "
     "exact.")

doc.add_heading("6.4 Explainability: class activation maps", 2)
para("A diagnosis is more useful, and more trustworthy, if the model can show where it looked. For a network "
     "that ends in global average pooling followed by a linear classifier, the class activation map (CAM) [11] "
     "for class c is:")
equation("CAMc(x, y)  =  Σk  wc,k · fk(x, y)")
para("Here fk(x, y) is the k-th feature map of the last convolutional layer and wc,k is the classifier weight "
     "connecting feature k to class c. Because the class score is exactly the spatial average of this map, "
     "the CAM is not an approximation: it shows precisely how much each region of the leaf contributed to the "
     "score. The map is upsampled to the image size and overlaid as a heatmap. If the bright region falls on "
     "the lesions, the model is using the right evidence; if it falls on the background, that is a warning sign.")

doc.add_heading("6.5 Results", 2)
para(f"On the held-out test set the model reaches {leaf_acc} accuracy and a macro-averaged F1 score of "
     f"{leaf_f1}. Macro F1 averages the F1 score of every class equally, so a rare class with poor results "
     f"would pull it down even if overall accuracy were high.")
if leaf and leaf.get("per_class"):
    rows = [[c.get("crop", ""), c.get("name", ""), f"{c['precision']:.3f}", f"{c['recall']:.3f}",
             f"{c['f1']:.3f}", c.get("support", "")] for c in leaf["per_class"]]
    table(["Crop", "Class", "Precision", "Recall", "F1", "Test images"], rows,
          [2.0, 6.4, 2.0, 1.8, 1.6, 2.4], bold_first_col=True)
else:
    para("[pending] Per-class precision, recall and F1 will appear here once results/leaf_metrics.json "
         "is produced.", italic=True, color=INK2)
para("Precision answers 'when the model says this disease, how often is it right?'; recall answers 'of all "
     "leaves with this disease, how many did the model catch?'.", size=10, color=INK2)
figure(RES / "fig_leaf_confusion.png", "Figure 7. Confusion matrix on the test set: rows are the true class, "
       "columns the predicted class. Off-diagonal cells are mistakes.", width_cm=15)
figure(RES / "fig_leaf_cam_examples.png", "Figure 8. Class activation maps for test images: the heatmap shows "
       "which regions of the leaf drove the predicted class.")

# ------------------------------------------------------------------ 7 website
doc.add_heading("7. Deployment: the AgriSense website", 1)
para(f"Both modules are deployed together as a public website: {SITE_URL}. It is a static site hosted free "
     f"on GitHub Pages. There is no server: everything is plain HTML, CSS and JavaScript that runs in the "
     f"visitor's browser. This makes the site free to host, fast, and private.")
table(["Part", "How it works", "Why"], [
    ["Irrigation simulator", "Every season (all cities, crops, sowing seasons, years and the 4 policies) is "
     "simulated in Python and shipped as JSON; the browser only draws the charts",
     "Identical numbers to this report; instant response; no Python needed online"],
    ["Agent rulebook", "The trained Q-tables are exported as JSON and shown as an interactive table",
     "The learned policy can be inspected on the site"],
    ["Leaf Doctor", f"The CNN is exported to ONNX ({leaf_size}) and executed by ONNX Runtime Web "
     "(WebAssembly) on the user's device; top-3 diagnoses, advice and a CAM heatmap are computed in the browser",
     "Photos never leave the device; works without a GPU or a server"],
    ["Local dashboard", "The Streamlit app (app.py) is kept for local use and live re-simulation",
     "Useful for development and experiments"],
], [3.0, 8.0, 5.4], bold_first_col=True)
para("The code is modular: the irrigation package (agrisense/), the leaf-disease training code (leaf/) and "
     "the website (index.html and web/) are separate, so a new crop, city, disease class or agent can be "
     "added without touching the rest.")

# ------------------------------------------------------------------ 8 limitations
doc.add_heading("8. Limitations and future work", 1)
doc.add_heading("8.1 Irrigation module", 2)
bullets([
    ("Simulated field. ", "Results come from an FAO-56 model, not a physical field trial; real soils vary "
     "within a field and sensors are noisy."),
    ("Simple yield model. ", "The seasonal FAO-33 equation ignores that stress at flowering hurts more than "
     "at other stages. A crop model such as AquaCrop would capture this."),
    ("Forecast. ", "The forecast is synthetic (true rain ± 25% error); real forecast archives would be more "
     "realistic."),
    ("Coarse state. ", "Ten dryness levels and three weather levels lose detail (e.g. Ludhiana winters). "
     "A Deep Q-Network over continuous inputs could remove this limit; the environment already "
     "supports it."),
    ("One soil. ", "Only sandy loam was trained; other soils would need retraining or soil as a state feature."),
])
doc.add_heading("8.2 Leaf disease module", 2)
bullets([
    ("Laboratory images. ", "PlantVillage photos show a single leaf on a plain background in good light. "
     "Real field photos (several leaves, soil, shadows, other symptoms) are harder, and accuracy on them "
     "will be lower than the test accuracy reported here."),
    ("Closed set of classes. ", f"The model only knows {leaf_n_classes} {leaf_crops} classes. Any other "
     "image, even a non-leaf, is forced into one of them; an 'unknown' or confidence-threshold option is "
     "needed in practice. Wheat, the third irrigation crop, is not in PlantVillage."),
    ("Coarse heatmaps. ", f"The CAM comes from the last feature grid ({leaf_grid} × {leaf_grid} cells), so it "
     "locates regions, not individual spots."),
    ("Next steps. ", "Fine-tune on field datasets such as PlantDoc, add stronger augmentation (backgrounds, "
     "lighting), add out-of-distribution rejection, and add more crops."),
])
doc.add_heading("8.3 System", 2)
bullets([
    "Connect the two modules: a detected disease could change the irrigation advice (for example, avoiding "
    "overhead watering when fungal blight is present).",
    "Feed real soil-moisture sensors and live forecasts into the irrigation agent for a field deployment.",
])

# ------------------------------------------------------------------ 9 conclusion
doc.add_heading("9. Conclusion", 1)
para(f"AgriSense applies two complementary AI techniques to one farmer's two biggest day-to-day questions: "
     f"when to water, and what is wrong with the crop. A simple, fully interpretable reinforcement learning "
     f"agent learned an irrigation policy from experience alone. On {n_test} unseen seasons across three very "
     f"different Indian climates and three crops, it used {save_fix:.0f}% less water than calendar-based "
     f"irrigation with a higher yield, and slightly outperformed an expert sensor rule in water use and water "
     f"productivity by learning to use the rain forecast. A transfer-learned MobileNetV2 recognises "
     f"{leaf_n_classes} healthy and diseased {leaf_crops} leaf classes with {leaf_acc} test accuracy, and "
     f"class activation maps show the evidence behind each diagnosis. Both run in a public website that "
     f"needs no server, so the work is not only a model but a usable, transparent tool.")

# ------------------------------------------------------------------ refs
doc.add_heading("References", 1)
for ref in [
    "[1] Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). Crop evapotranspiration: Guidelines for "
    "computing crop water requirements. FAO Irrigation and Drainage Paper 56. FAO, Rome.",
    "[2] Doorenbos, J., & Kassam, A. H. (1979). Yield response to water. FAO Irrigation and Drainage Paper 33. FAO, Rome.",
    "[3] Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. Machine Learning, 8(3-4), 279-292.",
    "[4] Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.",
    "[5] Zippenfenig, P. (2023). Open-Meteo.com Weather API. https://open-meteo.com (Historical Weather API, ERA5).",
    "[6] Hersbach, H., et al. (2020). The ERA5 global reanalysis. Quarterly Journal of the Royal "
    "Meteorological Society, 146(730), 1999-2049.",
    "[7] FAO (2019). New standards to curb the global spread of plant pests and diseases. Food and Agriculture "
    "Organization of the United Nations, news release, 3 April 2019.",
    "[8] Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the "
    "development of mobile disease diagnostics. arXiv:1511.08060.",
    "[9] Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). Using deep learning for image-based plant disease "
    "detection. Frontiers in Plant Science, 7, 1419.",
    "[10] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals "
    "and linear bottlenecks. Proc. IEEE/CVF CVPR, 4510-4520.",
    "[11] Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., & Torralba, A. (2016). Learning deep features for "
    "discriminative localization. Proc. IEEE CVPR, 2921-2929.",
    "[12] ONNX Runtime developers (2021). ONNX Runtime. https://onnxruntime.ai",
]:
    p = para(ref, size=10)
    p.paragraph_format.left_indent = Cm(0.8)
    p.paragraph_format.first_line_indent = Cm(-0.8)

# page numbers in footer
fp = sec.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = fp.add_run()
for tag, text in [("begin", None), (None, "PAGE"), ("end", None)]:
    if tag:
        el = OxmlElement("w:fldChar")
        el.set(qn("w:fldCharType"), tag)
    else:
        el = OxmlElement("w:instrText")
        el.set(qn("xml:space"), "preserve")
        el.text = text
    run._r.append(el)
sec.different_first_page_header_footer = True

doc.save(OUT)
print("saved", OUT)
