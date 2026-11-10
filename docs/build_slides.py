"""Builds docs/AgriSense_Presentation.pptx from the latest results.   python docs/build_slides.py

Story: problem -> need -> solution & system flowchart -> Module 1 (RL irrigation, benchmark, results)
-> Module 2 (leaf disease CNN: MobileNetV2 architecture, training, results) -> website -> demo
-> limitations -> conclusion.

Every slide has speaker notes (View > Notes in PowerPoint). Every chart has a "How to read it" and a
"What it shows" line that can be read aloud. Leaf numbers are read from results/leaf_metrics.json.
Inputs that do not exist yet are drawn as labelled placeholders, so re-running fills them in.
"""
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from agrisense.config import (CITIES, CROPS, DEFAULT_SOIL, IRRIGATION_ACTIONS_MM, SOILS,  # noqa: E402
                              RewardConfig, TrainConfig)

RES, DOCS = ROOT / "results", ROOT / "docs"
SCREENS = DOCS / "screens"
import os
OUT = Path(os.environ.get("DECK_OUT", DOCS / "AgriSense_Presentation.pptx"))
SITE_URL = "https://nirmit10000.github.io/agrisense/"

# palette: midnight + sand. Navy dominates dark slides, slate blue is the working accent,
# terracotta is reserved for key numbers, sand-gold lights up text on dark slides.
NAVY = RGBColor(0x1B, 0x2A, 0x41)
SLATE = RGBColor(0x3D, 0x5A, 0x80)
SLATE_TINT = RGBColor(0xE4, 0xEA, 0xF2)
SAND = RGBColor(0xF5, 0xF1, 0xEA)
SAND_LINE = RGBColor(0xE2, 0xD9, 0xCB)
TERRA = RGBColor(0xC8, 0x55, 0x3D)
TERRA_TINT = RGBColor(0xF8, 0xE6, 0xE0)
GOLD = RGBColor(0xE8, 0xC3, 0x9E)
INK = RGBColor(0x1B, 0x1F, 0x24)
MUTED = RGBColor(0x5B, 0x64, 0x72)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
POLICY = {"Q-learning agent (RL)": RGBColor(0x2A, 0x78, 0xD6), "Sensor threshold rule": RGBColor(0xEB, 0x68, 0x34),
          "Fixed schedule": RGBColor(0x1B, 0xAF, 0x7A), "Rainfed (no irrigation)": RGBColor(0xED, 0xA1, 0x00)}
HEAD, BODY = "Cambria", "Calibri"
# flowchart stage types: one colour each, used consistently on every flowchart
F_DATA = dict(fill=SAND, title_color=NAVY, sub_color=MUTED)
F_PROC = dict(fill=SLATE_TINT, title_color=NAVY, sub_color=MUTED)
F_MODEL = dict(fill=SLATE, title_color=WHITE, sub_color=WHITE)
F_OUT = dict(fill=NAVY, title_color=WHITE, sub_color=GOLD)

# ------------------------------------------------------------------ irrigation results
summary = pd.read_csv(RES / "summary_by_policy.csv", index_col=0)
evals = pd.read_csv(RES / "eval_results.csv")
RL, THR, FIX, RAIN = summary.index.tolist()
s_rl, s_thr, s_fix = summary.loc[RL], summary.loc[THR], summary.loc[FIX]
n_test = len(evals) // 4
save_fix = 100 * (1 - s_rl.avg_irrigation_mm / s_fix.avg_irrigation_mm)
save_thr = 100 * (1 - s_rl.avg_irrigation_mm / s_thr.avg_irrigation_mm)
rw, tc = RewardConfig(), TrainConfig()
maize = CROPS["Maize"]
taw_m = SOILS[DEFAULT_SOIL].taw(maize.root_depth_m)
raw_m = maize.p * taw_m
lud = evals[(evals.city == "Ludhiana") & (evals.policy == RL)].relative_yield.mean() * 100

# the example season (same as results/fig_season_trace.png)
from agrisense import weather as wx  # noqa: E402
from agrisense.agents import ThresholdPolicy  # noqa: E402
from agrisense.env import IrrigationEnv  # noqa: E402
from agrisense.simulate import all_policies, run_season  # noqa: E402
_trace = {}
for _pol in [all_policies("Maize")[0], ThresholdPolicy()]:
    _env = IrrigationEnv(wx.load("Jodhpur"), maize, DEFAULT_SOIL, seed=123 + 2023)
    _trace[_pol.name] = run_season(_env, _pol, _env.index_of("2023-06-15"))[0]
trace_rl, trace_thr = _trace[RL]["irrigation_mm"], _trace[THR]["irrigation_mm"]

# ------------------------------------------------------------------ leaf results (may change after retraining)
_leaf_path = RES / "leaf_metrics.json"
leaf = json.loads(_leaf_path.read_text(encoding="utf-8")) if _leaf_path.exists() else {}
DEFAULT_CLASSES = (
    [("Maize", n) for n in ["Gray leaf spot", "Common rust", "Northern leaf blight", "Healthy"]]
    + [("Tomato", n) for n in ["Bacterial spot", "Early blight", "Late blight", "Leaf mold", "Septoria leaf spot",
                               "Spider mites", "Target spot", "Yellow leaf curl virus", "Mosaic virus", "Healthy"]])
leaf_classes = [(c.get("crop", "?"), c.get("name", "?")) for c in leaf.get("classes", [])] or DEFAULT_CLASSES
n_classes = len(leaf_classes)
_acc, _f1 = leaf.get("test_accuracy"), leaf.get("macro_f1")
acc_txt = f"{100 * _acc:.1f} %" if isinstance(_acc, (int, float)) else "—"
f1_txt = f"{_f1:.3f}" if isinstance(_f1, (int, float)) else "—"
_n = leaf.get("n_images", {}) or {}
n_total = sum(v for v in _n.values() if isinstance(v, (int, float))) or None
n_test_img = _n.get("test")
_ep = leaf.get("epochs", {}) or {}
ep_head, ep_ft = _ep.get("head"), _ep.get("finetune")
train_min = leaf.get("train_minutes")
train_min_txt = f"{train_min:.0f} min" if isinstance(train_min, (int, float)) else "minutes"
input_size = leaf.get("input_size", 224)
ft_blocks = int(leaf.get("finetune_blocks", 4))
_pc = leaf.get("per_class") or []
weakest = sorted(_pc, key=lambda c: c.get("f1", 1))[:3]
weak_txt = ", ".join(f"{c.get('crop', '')[:3].lower()}. {c.get('name', '').lower()}" for c in weakest) or "similar spots"


def fmt_int(v):
    return f"{int(v):,}" if isinstance(v, (int, float)) else "—"


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = 13.333, 7.5


# ------------------------------------------------------------------ helpers
def text(slide, x, y, w, h, content, size=16, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, italic=False, spacing=None):
    """content: str or list of str/(str, dict) paragraphs; a paragraph may also be a list of runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = content if isinstance(content, list) else [content]
    for i, ptxt in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        runs = ptxt if isinstance(ptxt, list) else [ptxt]
        p.alignment = align
        if spacing:
            p.space_after = Pt(spacing)
        for rt in runs:
            opts = {}
            if isinstance(rt, tuple):
                rt, opts = rt
            if "align" in opts:
                p.alignment = opts["align"]
            r = p.add_run()
            r.text = rt
            f = r.font
            f.name = opts.get("font", font)
            f.size = Pt(opts.get("size", size))
            f.bold = opts.get("bold", bold)
            f.italic = opts.get("italic", italic)
            f.color.rgb = opts.get("color", color)
    return tb


def box(slide, x, y, w, h, fill=SAND, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line=None, radius=0.08, lw=1.25):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line:
        s.line.color.rgb = line
        s.line.width = Pt(lw)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    return s


def badge(slide, x, y, label, fill=SLATE, d=0.55, size=16, color=WHITE):
    """The deck's motif: a numbered/lettered circle."""
    c = box(slide, x, y, d, d, fill=fill, shape=MSO_SHAPE.OVAL)
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), True, color, BODY
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return c


def image(slide, path, x, y, w=None, h=None):
    """Place an image inside the (w, h) box, keeping aspect ratio, centred."""
    iw, ih = Image.open(path).size
    if w and h:
        scale = min(w / iw, h / ih)
        pw, ph = iw * scale, ih * scale
        x, y = x + (w - pw) / 2, y + (h - ph) / 2
    elif w:
        pw, ph = w, w * ih / iw
    else:
        pw, ph = h * iw / ih, h
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(pw), Inches(ph))


def picture(slide, path, x, y, w, h, label, frame=False, fill=False):
    """Image if the file exists, otherwise a tidy labelled placeholder of the same size."""
    if Path(path).exists():
        if fill:
            iw, ih = Image.open(path).size
            pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
            visible = (iw / ih) / (w / h)
            if visible < 1:
                pic.crop_bottom = 1 - visible
            else:
                pic.crop_right = 1 - 1 / visible
        else:
            pic = image(slide, path, x, y, w=w, h=h)
        if frame:
            pic.line.color.rgb = SAND_LINE
        return pic
    box(slide, x, y, w, h, fill=SAND, line=SAND_LINE, radius=0.04)
    text(slide, x + 0.2, y, w - 0.4, h, [label, (f"placeholder: {Path(path).name}", {"size": 11, "color": MUTED})],
         size=15, color=SLATE, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    return None


def explain(slide, x, y, w, how, what, size=12.5, h=0.95):
    """The read-aloud caption under every chart."""
    box(slide, x, y, w, h, fill=SAND, radius=0.1)
    text(slide, x + 0.2, y + 0.08, w - 0.4, h - 0.16, [
        [("How to read it:  ", {"bold": True, "color": SLATE}), (how, {"color": INK})],
        [("What it shows:  ", {"bold": True, "color": TERRA}), (what, {"color": INK})],
    ], size=size, anchor=MSO_ANCHOR.MIDDLE, spacing=3)


def _arrowhead(connector):
    from lxml import etree
    from pptx.oxml.ns import qn
    ln = connector.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("h", "med")


def line(slide, x1, y1, x2, y2, color=MUTED, width=2.0, head=False, dash=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    if dash:
        c.line.dash_style = 4
    if head:
        _arrowhead(c)
    return c


def arrow(slide, x1, y1, x2, y2, color=MUTED, width=2.25):
    return line(slide, x1, y1, x2, y2, color=color, width=width, head=True)


def node(slide, x, y, w, h, title, sub=None, fill=SAND, title_color=NAVY, sub_color=MUTED, size=14,
         sub_size=11.5, line_color=None):
    """Flowchart node: rounded box with a bold title and an optional smaller line."""
    box(slide, x, y, w, h, fill=fill, line=line_color, radius=0.12)
    paras = [(title, {"bold": True, "color": title_color, "size": size, "align": PP_ALIGN.CENTER})]
    if sub:
        paras.append((sub, {"color": sub_color, "size": sub_size, "align": PP_ALIGN.CENTER}))
    text(slide, x + 0.08, y + 0.05, w - 0.16, h - 0.1, paras, anchor=MSO_ANCHOR.MIDDLE, spacing=2)


def chain(slide, nodes, x0, y, w, h, gap, color=MUTED, **kw):
    """A horizontal row of nodes joined by arrows. nodes: list of (title, sub, style dict)."""
    for i, (title, sub, style) in enumerate(nodes):
        x = x0 + i * (w + gap)
        node(slide, x, y, w, h, title, sub, **style, **kw)
        if i:
            arrow(slide, x - gap + 0.04, y + h / 2, x - 0.04, y + h / 2, color=color, width=2)


def legend(slide, x, y, items, size=11.5):
    """Colour key for flowcharts: items = [(label, fill)]."""
    for i, (lbl, fill) in enumerate(items):
        xx = x + i * 2.35
        box(slide, xx, y + 0.04, 0.32, 0.22, fill=fill, radius=0.2, line=SAND_LINE if fill in (SAND, WHITE) else None)
        text(slide, xx + 0.42, y, 1.9, 0.3, lbl, size=size, color=MUTED)


def new_slide(title=None, dark=False, notes="", tag=None, tag_color=None):
    s = prs.slides.add_slide(BLANK)
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = NAVY if dark else WHITE
    if tag:
        text(s, 0.6, 0.22, 9, 0.3, tag, size=11, bold=True, color=tag_color or SLATE)
    if title:
        text(s, 0.6, 0.45, W - 1.2, 0.8, title, size=30, bold=True, font=HEAD, color=WHITE if dark else NAVY)
    s.notes_slide.notes_text_frame.text = notes or "(no notes)"
    num = len(prs.slides)
    if num > 1:
        text(s, W - 1.0, H - 0.42, 0.5, 0.3, str(num), size=10, color=GOLD if dark else MUTED, align=PP_ALIGN.RIGHT)
    return s


def stat(slide, x, y, w, value, label, value_color=NAVY, label_color=MUTED, size=54, label_size=14):
    text(slide, x, y, w, 1.0, value, size=size, bold=True, font=HEAD, color=value_color)
    text(slide, x, y + 1.0, w, 0.9, label, size=label_size, color=label_color)


M1 = "MODULE 1 · SMART IRRIGATION (REINFORCEMENT LEARNING)"
M2 = "MODULE 2 · LEAF DISEASE DETECTION (DEEP LEARNING)"
WEB = "DEPLOYMENT · THE WEBSITE"

# ================================================================== title
s = new_slide(dark=True, notes=(
    "Good morning. Our project is AgriSense: AI for smart irrigation and crop health. It falls under the course "
    "theme of agriculture: intelligent irrigation, crop monitoring and resource optimisation. It has two AI "
    "modules: a reinforcement learning agent that decides when and how much to irrigate, and a convolutional "
    "neural network that recognises leaf diseases from a photo. Both are deployed on one website."))
text(s, 0.9, 1.35, 11, 0.4, "EC-422 ARTIFICIAL INTELLIGENCE  ·  COURSE PROJECT", size=14, color=GOLD, bold=True)
text(s, 0.9, 1.95, 11, 1.4, "AgriSense", size=72, bold=True, font=HEAD, color=WHITE)
text(s, 0.9, 3.3, 11.5, 0.8, "AI for smart irrigation and crop health", size=30, color=WHITE, font=HEAD)
for i, lbl in enumerate(["Module 1  ·  Reinforcement learning decides when and how much to irrigate",
                         "Module 2  ·  A convolutional neural network diagnoses leaf diseases from a photo"]):
    badge(s, 0.9, 4.3 + i * 0.6, str(i + 1), fill=SLATE if i == 0 else TERRA, d=0.42, size=13)
    text(s, 1.5, 4.36 + i * 0.6, 10, 0.4, lbl, size=17, color=WHITE)
text(s, 0.9, 5.95, 8, 0.8, ["Dushyant Kumar  ·  23145030", "Nirmit Patil  ·  23145053"], size=16, color=WHITE)
text(s, 7.4, 6.2, 5.2, 0.4, SITE_URL.replace("https://", ""), size=16, color=GOLD, bold=True, align=PP_ALIGN.RIGHT)

# ================================================================== problem statement
s = new_slide("Problem statement: two decisions every farmer makes", notes=(
    "The course theme we picked is agriculture: intelligent irrigation, crop monitoring and resource "
    "optimisation. On a farm this comes down to two decisions that are still mostly made by habit. First, water: "
    "when should I irrigate, and how much? Second, crop health: this leaf looks wrong, is it a disease, and which "
    "one? Our problem statement is to support both decisions with AI."))
box(s, 0.6, 1.55, 12.1, 1.05, fill=NAVY)
text(s, 0.95, 1.55, 11.5, 1.05, [("Course theme", {"bold": True, "color": GOLD, "size": 14}),
                                ("Agriculture: intelligent irrigation, crop monitoring and resource optimisation",
                                 {"color": WHITE, "size": 20, "font": HEAD})], anchor=MSO_ANCHOR.MIDDLE)
for i, (b, col, tint, q, d) in enumerate([
        ("1", SLATE, SLATE_TINT, "Water: when, and how much, should I irrigate today?",
         "Decided by the calendar or by feel. Too much wastes water and drains away; too little stresses the crop."),
        ("2", TERRA, SAND, "Health: is this leaf diseased, and with what?",
         "Needs an expert who is rarely nearby. A late or wrong diagnosis means lost yield and wasted sprays.")]):
    x = 0.6 + i * 6.2
    box(s, x, 2.9, 5.9, 2.4, fill=tint)
    badge(s, x + 0.3, 3.15, b, fill=col, d=0.6, size=18)
    text(s, x + 1.1, 3.15, 4.6, 0.9, q, size=19, bold=True, color=NAVY)
    text(s, x + 1.1, 4.1, 4.5, 1.1, d, size=14, color=MUTED)
text(s, 0.6, 5.6, 12, 0.4, "Our problem statement", size=18, bold=True, color=NAVY, font=HEAD)
text(s, 0.6, 6.0, 12.1, 0.9, "Use AI to recommend daily irrigation that saves water without losing yield, and to "
     "diagnose maize and tomato leaf diseases from a photo, both usable from one website.", size=16, color=INK)

# ================================================================== the need
s = new_slide("Why it matters: water is wasted and crops are lost", notes=(
    "Why does this matter? Agriculture takes about 90 percent of India's freshwater withdrawals, according to FAO "
    "AQUASTAT. In our simulation of a typical fixed calendar, about 350 millimetres per season simply drains below "
    "the roots, because the calendar ignores rain, and the crop still loses yield in hot spells. On the crop-health "
    "side, the FAO estimates that up to 40 percent of global crop production is lost to plant pests and diseases "
    "every year. Early, correct diagnosis is the first step to preventing that."))
cards = [("~90 %", "of India's freshwater withdrawals go to agriculture", "FAO AQUASTAT", NAVY),
         (f"{s_fix.avg_wasted_mm:.0f} mm", "of irrigation per season drains away unused under a fixed 7-day calendar",
          f"our simulation, {n_test} test seasons", SLATE),
         ("40 %", "Up to this share of global crop production is lost to plant pests and diseases each year",
          "FAO", TERRA)]
for i, (v, l, src, col) in enumerate(cards):
    x = 0.6 + i * 4.1
    box(s, x, 1.6, 3.8, 3.3)
    text(s, x + 0.35, 1.85, 3.2, 1.0, v, size=50, bold=True, font=HEAD, color=col)
    text(s, x + 0.35, 2.95, 3.2, 1.2, l, size=16, color=INK)
    text(s, x + 0.35, 4.35, 3.2, 0.4, "Source: " + src, size=11, color=MUTED, italic=True)
box(s, 0.6, 5.25, 12.1, 1.5, fill=NAVY)
text(s, 0.95, 5.25, 11.4, 1.5, [("The gap", {"bold": True, "color": GOLD, "size": 14}),
                               ("Both decisions are made without data: irrigation by the calendar, diagnosis by eye. "
                                "One is a sequential decision problem, the other a visual recognition problem: both are "
                                "what AI is good at.", {"color": WHITE, "size": 18})],
     anchor=MSO_ANCHOR.MIDDLE, spacing=4)

# ================================================================== our solution
s = new_slide("Our solution: AgriSense, two AI modules", notes=(
    "AgriSense has two AI modules. Module one is a reinforcement learning agent: it looks at soil moisture, crop "
    "stage, the rain forecast and how hot the day is, and recommends an irrigation amount. Module two is a "
    "convolutional neural network: given a photo of a maize or tomato leaf, it returns the most likely disease with "
    "its confidence, a heatmap of where it looked, and management advice."))
mods = [("1", SLATE, SLATE_TINT, "Smart irrigation", "Reinforcement learning (tabular Q-learning)",
         ["Soil dryness, crop stage", "3-day rain forecast, evaporative demand"], "Irrigate 0–40 mm today",
         f"{save_fix:.0f} % less water than a fixed calendar"),
        ("2", TERRA, SAND, "Leaf disease detection", "Deep learning (MobileNetV2 CNN, transfer learning)",
         ["Photo of a maize or tomato leaf"], f"Top-3 of {n_classes} classes + heatmap + advice",
         f"{acc_txt} accuracy on unseen test images")]
for i, (b, col, tint, name, method, ins, out, res) in enumerate(mods):
    x = 0.6 + i * 6.2
    box(s, x, 1.55, 5.9, 5.2, fill=tint)
    badge(s, x + 0.3, 1.8, b, fill=col, d=0.65, size=20)
    text(s, x + 1.15, 1.78, 4.6, 0.45, name, size=22, bold=True, color=NAVY, font=HEAD)
    text(s, x + 1.15, 2.25, 4.6, 0.4, method, size=13, color=col, bold=True)
    for j, (lbl, body, style) in enumerate([("Input", ins, F_DATA), ("AI", [method.split(" (")[0]], F_MODEL),
                                            ("Output", [out], F_OUT)]):
        y = 2.95 + j * 0.95
        text(s, x + 0.35, y + 0.12, 1.0, 0.4, lbl, size=13, bold=True, color=MUTED)
        st = dict(style)
        if st["fill"] == SAND:
            st["fill"] = WHITE
        node(s, x + 1.35, y - 0.08, 4.25, 0.72, body[0], body[1] if len(body) > 1 else None, size=13, sub_size=12,
             **st)
        if j < 2:
            arrow(s, x + 3.47, y + 0.66, x + 3.47, y + 0.85, color=col, width=1.75)
    text(s, x + 0.35, 5.95, 5.3, 0.6, res, size=17, bold=True, color=col)

# ================================================================== system flowchart
s = new_slide("System architecture: train offline, run in the browser", notes=(
    "Here is the whole system as a flowchart. Column one is data: real weather from Open-Meteo with the FAO crop "
    "tables, and the PlantVillage leaf images. Column two is training, done offline in Python on a laptop: the "
    "Q-learning agent in our field simulator, and the MobileNetV2 network by transfer learning. Column three is "
    "what we export: the agent's Q-tables and all simulated seasons as JSON, and the network as an ONNX file. "
    "Column four is the website on GitHub Pages: it only serves files, and the visitor's browser does the "
    "computing, reading the JSON for irrigation and running the network with ONNX Runtime Web."))
cols = [("1", "Data", 0.55), ("2", "Train offline (Python)", 3.65), ("3", "Export as files", 6.75),
        ("4", "Serve and run", 9.85)]
for b, lbl, x in cols:
    badge(s, x, 1.45, b, fill=SLATE, d=0.45, size=13)
    text(s, x + 0.55, 1.5, 2.6, 0.4, lbl, size=15, bold=True, color=NAVY)
cw = 2.7
rows = [(2.15, SLATE, [("Weather + crop data", "Open-Meteo 2010-2025, 3 cities; FAO-56 crop tables", F_DATA),
                       ("Q-learning agent", "FAO-56 field simulator, 50,000 seasons per crop", F_MODEL),
                       ("Q-tables + seasons", "JSON: learned rulebook + every simulated season", F_PROC)]),
        (4.2, TERRA, [("Leaf images", f"PlantVillage, maize + tomato, {n_classes} classes", F_DATA),
                      ("MobileNetV2 CNN", "ImageNet-pretrained, fine-tuned (transfer learning)", F_MODEL),
                      ("Leaf model", "ONNX file + class list + advice", F_PROC)])]
for y, col, nodes in rows:
    for i, (t, sub, style) in enumerate(nodes):
        x = cols[i][2]
        node(s, x, y, cw, 1.55, t, sub, size=15, sub_size=12, **style)
        if i:
            arrow(s, x - 0.37, y + 0.78, x - 0.04, y + 0.78, color=col)
    arrow(s, cols[2][2] + cw + 0.04, y + 0.78, cols[3][2] - 0.04, y + 0.78, color=col)
box(s, cols[3][2], 2.15, 2.95, 3.6, fill=NAVY, radius=0.06)
text(s, cols[3][2] + 0.2, 2.25, 2.55, 3.4, [
    ("Static website", {"bold": True, "size": 18, "color": WHITE, "font": HEAD}),
    ("GitHub Pages", {"size": 13, "color": GOLD, "bold": True}),
    ("Smart irrigation: season simulator, rulebook, results", {"size": 13, "color": WHITE}),
    ("Leaf doctor: CNN in the browser (ONNX Runtime Web)", {"size": 13, "color": WHITE}),
], anchor=MSO_ANCHOR.MIDDLE, spacing=6)
legend(s, 0.6, 6.15, [("data", SAND), ("model / training", SLATE), ("exported artefact", SLATE_TINT),
                      ("deployed output", NAVY)])
text(s, 0.6, 6.6, 12.1, 0.4, "Inference is client-side: irrigation results are precomputed JSON; the CNN runs in "
     "WebAssembly via ONNX Runtime Web.", size=13, color=MUTED, italic=True)

# ================================================================== module 1 pipeline + why RL
s = new_slide("Module 1 · Smart irrigation with reinforcement learning", tag=M1, notes=(
    "Module one. Each day the agent reads the state of the field: soil dryness, crop growth stage, the rain "
    "forecast and evaporative demand. It looks up the best action in its learned Q-table and recommends an "
    "irrigation dose. During training, a field simulator works out what that dose does to the soil and the crop and "
    "returns a reward, which updates the table. Why reinforcement learning? Because irrigation is a sequence of "
    "decisions: today's watering changes tomorrow's soil, the yield is only known at the end of the season, rain is "
    "uncertain, and there is no labelled dataset of correct irrigation, so the agent learns from consequences."))
nodes = [("Weather + forecast", "Open-Meteo, real data", F_DATA),
         ("Field state", "soil dryness, crop stage, rain forecast, ET0", F_PROC),
         ("Q-table lookup", "360 states × 5 actions", F_MODEL),
         ("Irrigation dose", "0, 10, 20, 30 or 40 mm", F_OUT),
         ("Field simulator", "FAO-56 water balance", F_PROC)]
nw, gap, y0 = 2.2, 0.32, 1.65
chain(s, nodes, 0.6, y0, nw, 1.35, gap, color=SLATE, size=15, sub_size=12)
qx, sx = 0.6 + 2 * (nw + gap), 0.6 + 4 * (nw + gap)
yb = y0 + 1.35 + 0.45
line(s, sx + nw / 2, y0 + 1.35, sx + nw / 2, yb, color=TERRA, dash=True)
line(s, sx + nw / 2, yb, qx + nw / 2, yb, color=TERRA, dash=True)
arrow(s, qx + nw / 2, yb, qx + nw / 2, y0 + 1.37, color=TERRA, width=2)
text(s, qx + nw / 2 + 0.3, yb + 0.05, 5.5, 0.35, "reward + next state update the table (training only)", size=12,
     color=TERRA, italic=True)
text(s, 0.6, 4.05, 12, 0.45, "Why reinforcement learning?", size=20, bold=True, color=NAVY, font=HEAD)
reasons = [("Sequential", "today's watering changes tomorrow's soil moisture"),
           ("Delayed outcome", "yield is known only at the end of the season"),
           ("Uncertain", "rain forecasts are imperfect"),
           ("No labels", "no dataset of 'correct' irrigation, so it learns from consequences")]
for i, (h_, d) in enumerate(reasons):
    x = 0.6 + i * 3.1
    box(s, x, 4.65, 2.9, 2.05)
    badge(s, x + 0.25, 4.85, str(i + 1), fill=SLATE, d=0.5, size=14)
    text(s, x + 0.9, 4.9, 1.9, 0.45, h_, size=16, bold=True, color=NAVY)
    text(s, x + 0.25, 5.55, 2.5, 1.1, d, size=14, color=MUTED)

# ================================================================== data
s = new_slide("Data: 16 years of daily weather, three contrasting climates", tag=M1, notes=(
    "We used the Open-Meteo historical weather API, based on the ERA5 reanalysis: daily rainfall and reference "
    "evapotranspiration ET0, the drying power of the air, from 2010 to 2025. The left chart is average monthly "
    "rainfall, the right one monthly ET0. Pune has a heavy monsoon, Jodhpur is the driest with the highest ET0, "
    "Ludhiana gets some winter rain and has cold winters. Three different water regimes means one agent must "
    "generalise rather than memorise a place. Seasons starting 2010 to 2020 are for training, 2021 to 2024 for "
    "testing only."))
image(s, RES / "fig_climate.png", 0.5, 1.4, w=7.7, h=2.8)
explain(s, 0.6, 4.35, 7.5, "left: average rain per month; right: monthly ET0 (higher = drier air, more water "
        "needed). Lines are cities.", "three very different water regimes, so one agent must generalise across "
        "monsoon, desert and winter-rain climates.", h=1.15)
for i, c in enumerate(CITIES):
    w_ = pd.read_csv(ROOT / "data" / f"weather_{c.lower()}.csv", parse_dates=["date"])
    w_ = w_[w_.date.dt.year < 2025]
    rain = w_.groupby(w_.date.dt.year).rain_mm.sum().mean()
    y = 1.5 + i * 1.15
    box(s, 8.6, y, 4.2, 1.0)
    text(s, 8.85, y + 0.12, 2.0, 0.4, c, size=18, bold=True, color=NAVY)
    text(s, 10.75, y + 0.12, 1.85, 0.4, f"{rain:.0f} mm/yr", size=16, bold=True, color=SLATE, align=PP_ALIGN.RIGHT)
    text(s, 8.85, y + 0.5, 3.8, 0.4, CITIES[c]["climate"].split(";")[0], size=13, color=MUTED)
facts = [("Source", "Open-Meteo API (ERA5 reanalysis)"), ("Variables", "daily rain, ET0, Tmax, Tmin"),
         ("Train", "seasons starting 2010-2020"), ("Test", "seasons starting 2021-2024 (never seen)")]
for i, (k, v) in enumerate(facts):
    x = 0.6 + (i % 2) * 6.1
    y = 5.8 + (i // 2) * 0.55
    text(s, x, y, 1.4, 0.4, k, size=15, bold=True, color=NAVY)
    text(s, x + 1.4, y, 4.6, 0.4, v, size=15, color=INK)

# ================================================================== simulator
s = new_slide("The environment: a field simulator built on FAO-56", tag=M1, notes=(
    "We can't run 50 thousand seasons on a real farm, so the agent trains in a simulator built on FAO-56, the "
    "international standard for crop water requirements. Think of the root zone as a bucket. Rain and irrigation "
    "fill it; anything above the brim drains away and is wasted. The crop drinks from it every day: ET0 times a "
    "crop coefficient Kc that changes with growth stage. While the soil holds more than a safe amount, called RAW, "
    "the crop is fine; below that line it is water-stressed. At the end of the season the FAO-33 equation turns the "
    "total water shortfall into a yield loss."))
bx, by, bw, bh = 1.2, 2.5, 3.6, 3.3
box(s, bx, by, bw, bh, fill=SAND, shape=MSO_SHAPE.RECTANGLE, line=MUTED)
box(s, bx + 0.05, by + 1.3, bw - 0.1, bh - 1.35, fill=RGBColor(0xC9, 0xD6, 0xE6), shape=MSO_SHAPE.RECTANGLE)
line(s, bx, by + 1.95, bx + bw, by + 1.95, color=TERRA, width=2, dash=True)
text(s, bx + 0.15, by + 2.0, 3.3, 0.4, "stress line (RAW used up)", size=12, color=TERRA)
text(s, bx, by + bh + 0.1, bw, 0.4, "root zone (total available water, TAW)", size=12, color=MUTED,
     align=PP_ALIGN.CENTER)
arrow(s, bx + 0.7, 1.6, bx + 0.7, by + 0.5, color=SLATE, width=3)
text(s, bx - 0.1, 1.3, 1.8, 0.3, "rain + irrigation", size=13, color=SLATE, bold=True)
arrow(s, bx + 2.8, by + 0.5, bx + 2.8, 1.6, color=NAVY, width=3)
text(s, bx + 2.2, 1.3, 2.4, 0.3, "crop water use (ET)", size=13, color=NAVY, bold=True)
arrow(s, bx + bw, by + bh - 0.4, bx + bw + 1.0, by + bh - 0.4, color=MUTED, width=3)
text(s, bx + bw + 0.1, by + bh - 0.3, 1.6, 0.6, "overflow drains away (waste)", size=12, color=MUTED)
eqs = [("Crop demand", "ETc = Kc(stage) × ET0"),
       ("Stress factor", "Ks = 1 if depletion ≤ RAW, else falls linearly to 0"),
       ("Actual use", "ETa = Ks × ETc"),
       ("Season yield (FAO-33)", "Ya / Ym = 1 − Ky · (1 − ΣETa / ΣETc)")]
for i, (k, v) in enumerate(eqs):
    y = 1.7 + i * 1.15
    badge(s, 7.1, y, str(i + 1), fill=SLATE)
    text(s, 7.9, y - 0.05, 4.9, 0.4, k, size=16, bold=True, color=NAVY)
    text(s, 7.9, y + 0.35, 4.9, 0.5, v, size=16, color=INK, font="Cambria Math")
text(s, 7.1, 6.3, 5.8, 0.6, f"Maize on sandy loam: TAW = {taw_m:.0f} mm, RAW = p × TAW = {maize.p} × {taw_m:.0f} = "
     f"{raw_m:.0f} mm. Crop values from FAO-56 / FAO-33 tables.", size=12, color=MUTED)

# ================================================================== MDP
s = new_slide("Formulating irrigation as a Markov Decision Process", tag=M1, notes=(
    "The exact formulation. The state is four things a farmer would look at: how dry the soil is, the crop's growth "
    "stage, whether rain is forecast in the next three days, and how hot and dry today is. Discretised, that is 360 "
    "states. The actions are five irrigation amounts. The reward is negative: a cost for every millimetre of water, "
    "a ten times higher cost for every millimetre the crop wanted but didn't get, a cost for water that drains "
    "away, and a fixed cost per irrigation for pump and labour. The ten-to-one ratio encodes the priority that "
    "losing yield is far worse than spending water."))
mdp = [
    ("S", "State: what the agent sees", SLATE,
     ["Soil dryness: 10 levels", "Crop growth stage: 4", "Rain forecast, next 3 days: none / light / heavy",
      "Evaporative demand (ET0): low / medium / high", ("= 360 states", {"bold": True, "color": SLATE})]),
    ("A", "Action: what it decides", NAVY,
     [f"Irrigate {', '.join(map(str, IRRIGATION_ACTIONS_MM))} mm today", "Realistic drip / sprinkler doses",
      ("5 actions", {"bold": True, "color": NAVY})]),
    ("R", "Reward: how it is scored", TERRA,
     [f"− {rw.water_cost:g} per mm of water used", f"− {rw.stress_cost:g} per mm of crop water shortage",
      f"− {rw.waste_cost:g} per mm drained away", f"− {rw.event_cost:g} per irrigation event (pump, labour)",
      ("Yield matters 10× more than water", {"bold": True, "color": TERRA})]),
]
for i, (b, title, col, items) in enumerate(mdp):
    x = 0.6 + i * 4.1
    box(s, x, 1.6, 3.8, 4.85)
    badge(s, x + 0.3, 1.85, b, fill=col, d=0.65, size=20)
    text(s, x + 1.1, 1.9, 2.6, 0.6, title, size=17, bold=True, color=NAVY)
    text(s, x + 0.3, 2.8, 3.3, 3.5, items, size=15, color=INK, spacing=10)
text(s, 0.6, 6.65, 12, 0.4, "Episode = one crop season (125-145 days) from a random date, city and starting soil "
     "moisture. Forecast = real future rain ± 25 % error.", size=13, color=MUTED)

# ================================================================== Q-learning
s = new_slide("Learning: tabular Q-learning", tag=M1, notes=(
    "We used tabular Q-learning. The agent keeps a table with one value per state and action: how good is it, in "
    "the long run, to give this dose in this situation. After every simulated day it nudges that value towards the "
    "reward it got plus the discounted value of the best action in the next state; that is the update rule. Early "
    "on it explores randomly, with epsilon decaying from 1 to 0.05, and the learning rate also decays so the values "
    "settle. The chart shows the average season reward during training, one panel per crop. Reward is negative "
    "because every millimetre and every stress day costs something; higher is better. It climbs steadily while "
    "exploration decays and then flattens: the agent has converged. We chose a table rather than a deep network on "
    "purpose: 360 states is small, and a table can be printed and inspected."))
box(s, 0.6, 1.5, 6.1, 1.2, fill=SLATE_TINT)
text(s, 0.6, 1.5, 6.1, 1.2, "Q(s,a) ← Q(s,a) + α [ r + γ · max Q(s′,a′) − Q(s,a) ]", size=18,
     font="Cambria Math", color=NAVY, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
hp = [("Q-table", "360 states × 5 actions"), ("Episodes", f"{tc.episodes:,} seasons per crop"),
      ("Exploration ε", f"{tc.eps_start:g} → {tc.eps_end:g}"),
      ("Learning rate α", f"{tc.alpha_start:g} → {tc.alpha_end:g}"),
      ("Discount γ", f"{tc.gamma:g}"), ("Training time", "≈ 1.5 min per crop (CPU)")]
for i, (k, v) in enumerate(hp):
    y = 2.95 + i * 0.47
    text(s, 0.8, y, 2.4, 0.4, k, size=15, bold=True, color=NAVY)
    text(s, 3.2, y, 3.5, 0.4, v, size=15, color=INK)
text(s, 0.6, 5.85, 6.1, 1.0, [("Why a table, not a deep network?  ", {"bold": True, "color": NAVY}),
                             ("360 states is small enough to learn exactly, it trains in minutes on a laptop, and the "
                              "whole policy can be printed and checked.", {"color": INK})], size=14)
image(s, RES / "fig_training_curves.png", 7.0, 1.45, w=5.9, h=2.3)
explain(s, 7.0, 4.0, 5.9, "x = training seasons, y = total season reward (negative: every mm and stress day "
        "costs; higher = better). One panel per crop.", "reward climbs while exploration decays, then flattens: "
        "learning has converged for all three crops.", size=12.5, h=1.7)

# ================================================================== one day in the life
s = new_slide("How the agent decides: one day, step by step", tag=M1, notes=(
    "Here is one decision, step by step, so the algorithm is concrete. Step one: the agent observes the field. Say "
    "the soil is 70 to 80 percent dry, the maize is in mid-season, no rain is forecast for the next three days and "
    "the day is hot. That combination is one row of the Q-table. Step two: the row holds five numbers, one per dose. "
    "These values are illustrative, but the shape is real. They are all negative because they estimate the total "
    "future cost of the season. The highest, the least negative, is 40 millimetres. Step three: the agent "
    "irrigates 40 millimetres. Step four, during training only: the simulator moves the field to the next day and "
    "returns a reward of minus 50, which is 40 for the water plus 10 for the irrigation event, with no stress and "
    "no waste. Step five: the update rule nudges the Q-value towards that reward plus the discounted value of the "
    "best action tomorrow. Repeated millions of times, those small nudges become the rulebook."))
steps = [("1", "Observe", "soil 70–80 % dry · mid-season · rain forecast < 2 mm · ET0 high"),
         ("3", "Act", "irrigate 40 mm today"),
         ("4", "Simulator responds", "next day: soil ≈ 30 % dry · reward r = −(40 + 10) = −50"),
         ("5", "Learn (training only)", "Q ← Q + α [ r + γ · max Q(s′) − Q ]")]
node(s, 0.6, 1.6, 2.7, 1.5, "1 · Observe the state", steps[0][2], size=15, sub_size=12.5, **F_DATA)
arrow(s, 3.34, 2.35, 3.86, 2.35, color=SLATE)
# Q-table row (illustrative)
qv = [-612, -571, -540, -522, -509]
box(s, 3.9, 1.6, 5.2, 2.75, fill=SLATE_TINT, radius=0.06)
text(s, 4.1, 1.68, 4.8, 0.35, "2 · Look up this row of the Q-table", size=15, bold=True, color=NAVY)
for i, (a, v) in enumerate(zip(IRRIGATION_ACTIONS_MM, qv)):
    x = 4.1 + i * 0.97
    best = i == len(qv) - 1
    box(s, x, 2.15, 0.88, 1.15, fill=SLATE if best else WHITE, radius=0.12, line=None if best else SAND_LINE)
    text(s, x, 2.2, 0.88, 0.45, f"{a} mm", size=13, bold=True, color=WHITE if best else MUTED, align=PP_ALIGN.CENTER)
    text(s, x, 2.62, 0.88, 0.55, f"{v}", size=17, bold=True, font=HEAD, color=WHITE if best else NAVY,
         align=PP_ALIGN.CENTER)
text(s, 4.1, 3.42, 4.8, 0.8, [[("Pick the highest value → 40 mm.  ", {"bold": True, "color": NAVY}),
                              ("Values are illustrative: each is the expected total future reward of the season.",
                               {"color": MUTED, "italic": True})]], size=12.5)
arrow(s, 9.14, 2.35, 9.66, 2.35, color=SLATE)
node(s, 9.7, 1.6, 3.0, 1.5, "3 · Act", "irrigate 40 mm today", size=15, sub_size=13, **F_OUT)
arrow(s, 11.2, 3.14, 11.2, 4.56, color=TERRA)
node(s, 9.7, 4.6, 3.0, 1.55, "4 · Simulator responds", "next day soil ≈ 30 % dry; reward r = −(40 water + 10 event) "
     "= −50", size=15, sub_size=12.5, **F_PROC)
arrow(s, 9.66, 5.37, 7.84, 5.37, color=TERRA)
box(s, 0.6, 4.6, 7.2, 1.55, fill=SAND, radius=0.08)
text(s, 0.85, 4.66, 6.8, 1.45, [
    ("5 · Learn: nudge the Q-value (training only)", {"bold": True, "color": NAVY, "size": 15}),
    ("target = r + γ · max Q(s′) = −50 + 0.95 × (−480) = −506", {"font": "Cambria Math", "size": 14}),
    ("Q ← −509 + α · (−506 − (−509)) = −509 + 0.05 × 3 ≈ −508.9", {"font": "Cambria Math", "size": 14}),
], anchor=MSO_ANCHOR.MIDDLE, spacing=3)
text(s, 0.6, 6.4, 12.1, 0.55, "Repeated for every day of 50,000 simulated seasons, these small nudges turn the table "
     "into the rulebook shown later. After training only steps 1–3 are used.", size=13, color=MUTED, italic=True)

# ================================================================== the benchmark: sensor threshold rule
s = new_slide("The benchmark: FAO's soil-moisture threshold rule", tag=M1, notes=(
    "Beating a naive calendar is easy, so we also compare against a strong expert benchmark: the FAO-56 "
    "management rule. With a soil-moisture sensor, irrigate as soon as the depletion passes the readily available "
    f"water, RAW, which for maize on our soil is {raw_m:.0f} millimetres, and refill the root zone exactly to field "
    "capacity. We even give it advantages our agent does not have: a perfect sensor and any irrigation amount, not "
    "just five doses. It never stresses the crop, so it gets 100 percent yield. Its blind spot is that it ignores "
    "the forecast and always fills the soil to the brim, so rain that follows overflows. Against it, our agent uses "
    f"{save_thr:.1f} percent less water and wastes less, for {100 - s_rl.avg_relative_yield_pct:.1f} points of yield. "
    "The important point: the agent was never told this rule, yet reached the same level by trial and error."))
box(s, 0.6, 1.55, 5.6, 2.35, fill=NAVY, radius=0.06)
text(s, 0.9, 1.65, 5.1, 2.15, [
    ("The rule (FAO-56 management-allowed depletion)", {"bold": True, "color": GOLD, "size": 14}),
    ("if  depletion  >  RAW = p × TAW", {"color": WHITE, "size": 18, "font": "Cambria Math"}),
    ("    irrigate depletion mm  (refill to field capacity)", {"color": WHITE, "size": 16, "font": "Cambria Math"}),
    ("else  do nothing", {"color": WHITE, "size": 18, "font": "Cambria Math"}),
    (f"maize: TAW = {taw_m:.0f} mm, p = {maize.p}, so it irrigates once {raw_m:.0f} mm is used",
     {"color": GOLD, "size": 12.5}),
], anchor=MSO_ANCHOR.MIDDLE, spacing=4)
for i, (b, col, tint, h_, d) in enumerate([
        ("+", SLATE, SLATE_TINT, "Why it is a strong benchmark",
         "Recommended expert practice; we give it a perfect soil sensor and any dose size, so it never stresses "
         "the crop (100 % yield)."),
        ("−", TERRA, TERRA_TINT, "Its blind spot",
         "Ignores the forecast and always refills to 100 %, so rain right after an irrigation overflows and is lost.")]):
    y = 4.1 + i * 1.45
    box(s, 0.6, y, 5.6, 1.3, fill=tint)
    badge(s, 0.8, y + 0.2, b, fill=col, d=0.5, size=18)
    text(s, 1.45, y + 0.12, 4.6, 0.4, h_, size=15, bold=True, color=NAVY)
    text(s, 1.45, y + 0.5, 4.6, 0.8, d, size=12.5, color=INK)
text(s, 6.6, 1.55, 6.2, 0.4, f"RL agent vs the benchmark ({n_test} unseen seasons)", size=17, bold=True, color=NAVY,
     font=HEAD)
rows_ = [("", "RL agent", "Sensor rule"),
         ("Water per season", f"{s_rl.avg_irrigation_mm:.0f} mm", f"{s_thr.avg_irrigation_mm:.0f} mm"),
         ("Wasted to drainage", f"{s_rl.avg_wasted_mm:.1f} mm", f"{s_thr.avg_wasted_mm:.1f} mm"),
         ("Irrigation events", f"{s_rl.avg_irrigation_events:.1f}", f"{s_thr.avg_irrigation_events:.1f}"),
         ("Yield (% of max)", f"{s_rl.avg_relative_yield_pct:.1f} %", f"{s_thr.avg_relative_yield_pct:.1f} %"),
         ("Crop per m³ of water", f"{s_rl.water_productivity_kg_per_m3:.2f} kg",
          f"{s_thr.water_productivity_kg_per_m3:.2f} kg"),
         ("Knows the FAO rule?", "no, learned it", "yes, hand-coded")]
for r, (a, b, c) in enumerate(rows_):
    y = 2.1 + r * 0.52
    if r:
        box(s, 6.6, y - 0.04, 6.2, 0.46, fill=SAND if r % 2 else WHITE, radius=0.15)
    hdr = r == 0
    text(s, 6.8, y, 2.6, 0.4, a, size=14, bold=True, color=NAVY)
    text(s, 9.4, y, 1.6, 0.4, b, size=14, bold=hdr or r in (1, 5), color=POLICY[RL] if hdr else INK,
         align=PP_ALIGN.CENTER)
    text(s, 11.1, y, 1.6, 0.4, c, size=14, bold=hdr, color=POLICY[THR] if hdr else INK, align=PP_ALIGN.CENTER)
box(s, 6.6, 5.85, 6.2, 0.95, fill=NAVY, radius=0.1)
text(s, 6.85, 5.85, 5.8, 0.95, [[("Takeaway  ", {"bold": True, "color": GOLD}),
                                (f"expert-level behaviour learned from scratch, plus {save_thr:.1f} % less water by "
                                 "using the forecast.", {"color": WHITE})]], size=14, anchor=MSO_ANCHOR.MIDDLE)

# ================================================================== fair test
s = new_slide("A fair test: four policies on seasons the agent never saw", tag=M1, notes=(
    "We compare four policies. Rainfed never irrigates, the lower bound. Fixed schedule is the typical farmer "
    "calendar, 40 millimetres every 7 days. The sensor threshold rule is the expert benchmark we just saw. And our "
    f"agent. All four run on the same {n_test} test seasons from 2021 to 2024: three cities, five crop-season "
    "combinations, four years, with identical weather and identical forecast errors."))
pol = [(RL, "Our agent. Uses soil, crop stage, forecast and ET0."),
       (THR, "FAO expert rule. Perfect moisture sensor; refills to full at the safe limit; no forecast."),
       (FIX, "Farmer's calendar: 40 mm every 7 days, whatever the weather."),
       (RAIN, "Never irrigate. Lower bound.")]
for i, (name, desc) in enumerate(pol):
    x = 0.6 + (i % 2) * 4.3
    y = 1.6 + (i // 2) * 2.35
    box(s, x, y, 4.0, 2.1)
    badge(s, x + 0.3, y + 0.3, "", fill=POLICY[name], d=0.4)
    text(s, x + 0.9, y + 0.28, 3.0, 0.5, name, size=17, bold=True, color=NAVY)
    text(s, x + 0.3, y + 0.9, 3.5, 1.2, desc, size=14, color=MUTED)
box(s, 9.5, 1.6, 3.3, 4.45, fill=NAVY)
text(s, 9.8, 1.9, 2.8, 1.0, str(n_test), size=60, bold=True, font=HEAD, color=GOLD)
text(s, 9.8, 2.95, 2.8, 3.0, ["unseen test seasons", "", "3 cities × 5 crop-season combinations × 4 years "
                              "(2021-2024)", "", "Same weather and forecast errors for every policy"],
     size=14, color=WHITE)
text(s, 0.6, 6.4, 12, 0.5, "Metrics: irrigation water · yield (% of max) · water wasted · stress days · "
     "water productivity (kg of crop per m³ of water)", size=14, color=MUTED)

# ================================================================== irrigation results
s = new_slide(f"Result: {save_fix:.0f} % less water than the calendar, higher yield", tag=M1, notes=(
    f"The headline numbers, averaged over all {n_test} test seasons. The left bars are water used per season, the "
    "right bars yield as a percentage of the maximum. Compared with the fixed calendar, the agent uses "
    f"{save_fix:.0f} percent less water, about {s_rl.avg_irrigation_mm:.0f} millimetres instead of "
    f"{s_fix.avg_irrigation_mm:.0f}, and the yield goes up, from {s_fix.avg_relative_yield_pct:.1f} to "
    f"{s_rl.avg_relative_yield_pct:.1f} percent. Against the expert sensor rule it uses {save_thr:.1f} percent less "
    "water for a small yield trade-off, and it produces the most crop per cubic metre of water of any policy."))
stats_ = [(f"{save_fix:.0f} %", "less water than the fixed calendar", TERRA),
          (f"{s_rl.avg_relative_yield_pct:.1f} %", f"of maximum yield (calendar: {s_fix.avg_relative_yield_pct:.1f} %)",
           NAVY),
          (f"{s_rl.water_productivity_kg_per_m3:.2f}", f"kg of crop per m³ of water, the best of all "
                                                       f"(sensor rule {s_thr.water_productivity_kg_per_m3:.2f})", NAVY)]
for i, (v, l, c) in enumerate(stats_):
    stat(s, 0.6, 1.45 + i * 1.75, 3.4, v, l, value_color=c, size=46)
image(s, RES / "fig_policy_comparison.png", 4.3, 1.45, w=8.6, h=3.1)
explain(s, 4.4, 4.8, 8.4, f"left: water used per season; right: yield as % of the maximum; averages over {n_test} "
        "unseen seasons, one colour per policy.", "the agent uses about the expert rule's water and far less than "
        "the calendar, while keeping yield above 98 %.", h=1.3)

# ================================================================== climates
s = new_slide("One agent adapts to very different climates", tag=M1, notes=(
    "This chart breaks the water use down by crop and city. Each group of bars is one crop in one city; the bar "
    "height is the average water per season for each policy. Water needs differ by about a factor of three "
    "between climates, yet one agent per crop handles all of them; it is never told which city it is in. It always "
    "uses much less than the calendar and slightly less than the sensor rule. Its weakest city is Ludhiana, where "
    f"yield drops to about {lud:.0f} percent, most likely because the cold-winter states are coarse in our table."))
image(s, RES / "fig_water_by_city.png", 0.5, 1.45, w=12.3, h=3.7)
explain(s, 0.6, 5.35, 12.1, "one panel per crop; inside it, one group per city; bar height = average irrigation "
        "water per season for each policy (blue = RL, orange = sensor rule, green = fixed calendar).",
        f"needs vary about 3× between climates and the agent adapts without knowing the city; weakest is "
        f"Ludhiana (yield {lud:.1f} %).", h=1.3)

# ================================================================== one season
s = new_slide("One season: smaller doses that leave room for rain", tag=M1, notes=(
    "This is one real test season: maize in Jodhpur, monsoon 2023. The top panel is the water available in the "
    "soil; the dashed line is where stress begins. The middle panel is rain and the bottom panel the irrigation "
    "events. The orange sensor rule waits until the line is reached, then fills the soil to the top with about 55 "
    "millimetres. The blue agent gives smaller doses of 30 to 40 millimetres, which leaves room for rain. That "
    f"season it used {trace_rl:.0f} millimetres against {trace_thr:.0f} for the sensor rule, "
    f"{100 * (1 - trace_rl / trace_thr):.0f} percent less."))
image(s, RES / "fig_season_trace.png", 0.6, 1.35, w=7.6, h=5.3)
notes_ = [("Sensor rule", "waits for the stress line, then refills to 100 %, so later rain can overflow", POLICY[THR]),
          ("RL agent", "smaller 30-40 mm doses keep room in the soil for rain", POLICY[RL]),
          ("This season", f"{trace_rl:.0f} mm vs {trace_thr:.0f} mm: {100 * (1 - trace_rl / trace_thr):.0f} % "
                          "less water", TERRA)]
for i, (h_, d, col) in enumerate(notes_):
    y = 1.45 + i * 1.05
    badge(s, 8.6, y, str(i + 1), fill=col, d=0.5, size=14)
    text(s, 9.25, y - 0.02, 3.6, 0.4, h_, size=16, bold=True, color=NAVY)
    text(s, 9.25, y + 0.36, 3.6, 0.7, d, size=13, color=MUTED)
explain(s, 8.6, 4.75, 4.2, "top: soil water %, dashed = stress line; middle: rain; bottom: irrigation (blue = RL, "
        "orange = sensor rule).", "both keep the crop above the stress line; the agent does it with smaller, "
        "earlier doses and less water.", size=12, h=1.9)

# ================================================================== rulebook
s = new_slide("What the agent learned can be read like a rulebook", tag=M1, notes=(
    "Because the policy is a table, we can open it. This is maize on a hot day, one panel per growth stage. Rows are "
    "soil dryness, columns the rain forecast, and the number in each cell is the dose the agent picks; darker means "
    "more water. Three rules emerge that nobody programmed: do nothing until about 60 percent of the soil water is "
    "used, close to the FAO threshold of 55 percent; when it's dry and no rain is coming, give the full 40 "
    "millimetres; and when heavy rain is forecast, give less or nothing. Cells marked rare were seen fewer than 50 "
    "times in training, so we don't read anything into them."))
image(s, RES / "fig_policy_heatmap.png", 0.5, 1.4, w=12.3, h=4.0)
explain(s, 0.6, 5.5, 12.1, "one panel per growth stage; rows = soil dryness, columns = rain forecast; each cell = "
        "the dose chosen (mm), darker = more; 'rare' = seen < 50 times.", "no water until ~60 % dry (FAO says 55 %), "
        "full 40 mm when dry with no rain coming, less when heavy rain is forecast.", h=1.3)

# ================================================================== module 2 pipeline + why CNN
s = new_slide("Module 2 · Leaf disease detection with a CNN", tag=M2, tag_color=TERRA, notes=(
    "Module two diagnoses leaf diseases. The pipeline: a photo of a leaf is resized and centre-cropped to 224 by 224 "
    "pixels and normalised. A MobileNetV2 convolutional network, pretrained on ImageNet, turns it into 1280 feature "
    "maps of 7 by 7 describing textures, spots and colours. Global average pooling squeezes each map to one number, "
    f"and a linear classifier maps those 1280 numbers to our {n_classes} classes. Softmax gives probabilities; we "
    "show the top three, plus a class activation heatmap computed from the same feature maps. Why a CNN? Disease is "
    "a visual pattern, and convolutions detect local patterns anywhere in the image. Why transfer learning? From "
    "scratch we would need millions of images; starting from ImageNet weights, a few thousand leaf photos suffice."))
nodes = [("Leaf photo", "phone or upload", F_DATA),
         ("Pre-process", f"resize, crop {input_size}², normalise", F_PROC),
         ("MobileNetV2", "7×7×1280 feature maps", F_MODEL),
         ("Global avg pool", "1280 numbers", F_PROC),
         ("Linear layer", f"1280 → {n_classes}", F_MODEL),
         ("Softmax", "probabilities", F_PROC),
         ("Diagnosis", "top-3, heatmap, advice", F_OUT)]
nw, gap = 1.52, 0.27
chain(s, nodes, 0.55, 1.55, nw, 1.45, gap, color=SLATE, size=14, sub_size=11.5)
x_mb, x_lin = 0.55 + 2 * (nw + gap), 0.55 + 4 * (nw + gap)
line(s, x_mb + nw / 2, 3.0, x_mb + nw / 2, 3.35, color=TERRA, dash=True)
line(s, x_mb + nw / 2, 3.35, x_lin + nw / 2, 3.35, color=TERRA, dash=True)
text(s, x_mb + nw / 2 + 0.1, 3.4, 6.5, 0.35, "feature maps × classifier weights = class activation heatmap",
     size=12.5, color=TERRA, italic=True)
legend(s, 0.6, 3.85, [("input", SAND), ("processing", SLATE_TINT), ("learned layers", SLATE), ("output", NAVY)])
text(s, 0.6, 4.25, 12, 0.45, "Why a CNN, and why transfer learning?", size=20, bold=True, color=NAVY, font=HEAD)
whys = [("Visual pattern", "disease shows as spots, rings and colour changes: local patterns that convolutions detect"),
        ("Weight sharing", "the same filter scans the whole leaf, so a lesion is found wherever it is"),
        ("Transfer learning", "ImageNet weights already know edges and textures; thousands of leaf photos suffice"),
        ("Explainable", "global pooling + one linear layer lets us draw an exact heatmap")]
for i, (h_, d) in enumerate(whys):
    x = 0.6 + i * 3.1
    box(s, x, 4.8, 2.9, 1.95)
    badge(s, x + 0.25, 4.98, str(i + 1), fill=TERRA, d=0.5, size=14)
    text(s, x + 0.9, 5.03, 1.95, 0.5, h_, size=15.5, bold=True, color=NAVY)
    text(s, x + 0.25, 5.6, 2.5, 1.1, d, size=13, color=MUTED)

# ================================================================== leaf data
s = new_slide("Data: PlantVillage leaf images, maize and tomato", tag=M2, tag_color=TERRA, notes=(
    f"For data we used PlantVillage, the standard public benchmark of labelled leaf photos. We kept only maize and "
    f"tomato, the crops our irrigation module also covers: {n_classes} classes including the healthy ones. The chart "
    "shows how many images per class we used: we capped each class at 600 so no class dominates, and only two "
    f"classes have fewer. That is {fmt_int(n_total)} images in total, split 70, 15, 15 into training, validation and "
    "test; the test images are never seen during training. Wheat is not in PlantVillage."))
picture(s, RES / "fig_leaf_dataset.png", 0.55, 1.4, 6.7, 3.5, "Images per class")
explain(s, 0.6, 5.05, 6.6, "one bar per class (maize green, tomato orange); length = number of images used.",
        "classes are nearly balanced (capped at 600), so accuracy is not dominated by one class.", size=12, h=0.95)
crops_ = {}
for crop, name in leaf_classes:
    crops_.setdefault(crop, []).append(name)
for j, (crop, names) in enumerate(crops_.items()):
    col_x = 7.6 + (0 if j == 0 else 2.55)
    text(s, col_x, 1.45, 2.4, 0.4, f"{crop} ({len(names)})", size=17, bold=True, color=NAVY)
    text(s, col_x, 1.9, 2.45, 3.8, [(n, {"color": SLATE if n.lower().startswith("healthy") else INK,
                                         "bold": n.lower().startswith("healthy")}) for n in names],
         size=12.5, spacing=3)
splits = [("Total images", fmt_int(n_total)), ("Train", fmt_int(_n.get("train"))),
          ("Validation", fmt_int(_n.get("val"))), ("Test (unseen)", fmt_int(_n.get("test")))]
for i, (k, v) in enumerate(splits):
    x = 0.6 + i * 3.1
    box(s, x, 6.15, 2.9, 0.85, fill=SLATE_TINT)
    text(s, x + 0.25, 6.17, 1.5, 0.45, v, size=21, bold=True, color=NAVY, font=HEAD)
    text(s, x + 1.55, 6.32, 1.3, 0.35, k, size=12.5, color=MUTED)

# ================================================================== MobileNetV2 architecture
s = new_slide("Inside the model: the MobileNetV2 architecture", tag=M2, tag_color=TERRA, notes=(
    "This is the network we used, MobileNetV2. A 224 by 224 colour image goes through one ordinary 3 by 3 "
    "convolution with 32 filters, then 17 inverted-residual bottleneck blocks, grouped into seven stages. Each stage "
    "halves the image size or keeps it, and increases the number of channels: from 112 by 112 down to 7 by 7, and "
    "from 16 up to 320 channels. Early stages detect edges and colours, later ones textures and lesion shapes. A "
    "final 1 by 1 convolution produces 1280 feature maps of 7 by 7. Global average pooling turns each map into one "
    f"number, and our new linear layer maps the 1280 numbers to {n_classes} disease classes. The colours show how we "
    f"trained it: the early part keeps its ImageNet weights frozen, the last {ft_blocks} blocks were fine-tuned on "
    "leaves, and the classifier is new."))
FEAT = [("Conv 3×3", "32 ch", "112×112", [0]), ("Stage 1", "16 ch · 1 block", "112×112", [1]),
        ("Stage 2", "24 ch · 2 blocks", "56×56", [2, 3]), ("Stage 3", "32 ch · 3 blocks", "28×28", [4, 5, 6]),
        ("Stage 4", "64 ch · 4 blocks", "14×14", [7, 8, 9, 10]), ("Stage 5", "96 ch · 3 blocks", "14×14", [11, 12, 13]),
        ("Stage 6", "160 ch · 3 blocks", "7×7", [14, 15, 16]), ("Stage 7", "320 ch · 1 block", "7×7", [17]),
        ("Conv 1×1", "1280 ch", "7×7", [18])]
ft_set = set(range(19 - ft_blocks, 19))
HEIGHTS = {"112×112": 2.4, "56×56": 2.05, "28×28": 1.75, "14×14": 1.45, "7×7": 1.15}
x, mid, bw_ = 0.6, 3.1, 1.12
node(s, x, mid - 1.4, 0.95, 2.8, "Input", "224×224×3 leaf photo", size=13, sub_size=11, **F_DATA)
x += 0.95 + 0.17
for name, ch, res, idx in FEAT:
    hh = HEIGHTS[res]
    n_ft = len(set(idx) & ft_set)
    if n_ft == len(idx):
        style, tag_ = F_MODEL, "fine-tuned"
    elif n_ft:
        style, tag_ = dict(fill=RGBColor(0x8F, 0xA5, 0xC2), title_color=WHITE, sub_color=WHITE), \
            f"last {n_ft} fine-tuned"
    else:
        style, tag_ = F_PROC, None
    node(s, x, mid - hh / 2, bw_, hh, name, f"{ch}\n{res}", size=12.5, sub_size=10.5, **style)
    arrow(s, x - 0.16, mid, x - 0.02, mid, color=MUTED, width=1.5)
    x += bw_ + 0.17
# bracket over the 17 bottleneck blocks
bx0, bx1 = 0.6 + 0.95 + 0.17 + (bw_ + 0.17), 0.6 + 0.95 + 0.17 + 8 * (bw_ + 0.17) - 0.17
line(s, bx0, 1.68, bx1, 1.68, color=SLATE, width=1.5)
line(s, bx0, 1.68, bx0, 1.82, color=SLATE, width=1.5)
line(s, bx1, 1.68, bx1, 1.82, color=SLATE, width=1.5)
text(s, bx0, 1.36, bx1 - bx0, 0.3, "17 inverted-residual bottleneck blocks (detail on next slide)", size=12,
     color=SLATE, bold=True, align=PP_ALIGN.CENTER)
# head row
yh = 4.85
head = [("Feature maps", "7×7×1280", dict(fill=SLATE, title_color=WHITE, sub_color=WHITE)),
        ("Global avg pool", "→ 1280 numbers", F_PROC),
        ("Linear layer", f"1280 → {n_classes} (new)", dict(fill=TERRA, title_color=WHITE, sub_color=WHITE)),
        ("Softmax", f"{n_classes} probabilities", F_OUT)]
hw, hg = 1.6, 0.28
x_last = 0.6 + 0.95 + 0.17 + 9 * (bw_ + 0.17) - 0.17
hx = x_last - 4 * hw - 3 * hg
xc = x_last - bw_ / 2  # centre of the final 1x1 conv: route its output down and left into the head row
line(s, xc, mid + 0.6, xc, yh - 0.3, color=MUTED, width=1.5)
line(s, xc, yh - 0.3, hx + hw / 2, yh - 0.3, color=MUTED, width=1.5)
arrow(s, hx + hw / 2, yh - 0.3, hx + hw / 2, yh - 0.02, color=MUTED, width=1.5)
for i, (t, sub, st) in enumerate(head):
    xx = hx + i * (hw + hg)
    node(s, xx, yh, hw, 0.95, t, sub, size=13, sub_size=11.5, **st)
    if i:
        arrow(s, xx - hg + 0.03, yh + 0.47, xx - 0.03, yh + 0.47, color=MUTED, width=1.5)
legend(s, 0.6, 4.9, [("pretrained, frozen", SLATE_TINT), ("fine-tuned on leaves", SLATE)], size=12)
legend(s, 0.6, 5.35, [("new layer we trained", TERRA), ("input / output", NAVY)], size=12)
box(s, 0.6, 6.0, 12.15, 0.9, fill=SAND, radius=0.1)
text(s, 0.85, 6.0, 11.7, 0.9, [[("In numbers:  ", {"bold": True, "color": NAVY}),
                                ("~3.5 M parameters, ~300 M multiply-adds per image; image shrinks 224 → 7 while "
                                 "channels grow 3 → 1280; we fine-tuned the last " + str(ft_blocks) + " of 19 feature "
                                 "blocks and trained a new 14-way classifier.", {"color": INK})]],
     size=13.5, anchor=MSO_ANCHOR.MIDDLE)

# ================================================================== inverted residual block
s = new_slide("The building block: an inverted residual bottleneck", tag=M2, tag_color=TERRA, notes=(
    "Each of those 17 blocks has the same three steps. First a 1 by 1 convolution expands the channels six times, "
    "say from 24 to 144. Then a 3 by 3 depthwise convolution filters each channel separately, looking at the "
    "neighbourhood of every pixel. Then a 1 by 1 linear projection squeezes the channels back down, without a ReLU, "
    "because a ReLU in a narrow layer would destroy information. When the input and output have the same shape, a "
    "skip connection adds the input back, which makes deep networks easier to train. It is called inverted because "
    "a classic ResNet bottleneck goes wide, narrow, wide; this one goes narrow, wide, narrow. The trick that makes it "
    "cheap is the depthwise-separable convolution: splitting a normal 3 by 3 convolution into a per-channel 3 by 3 "
    "and a 1 by 1 mix needs about 8 to 9 times fewer multiplications."))
y_c = 3.1
blocks = [("Input", "h × w × c", 0.55, F_DATA), ("1×1 conv", "expand × 6\nReLU6", 1.6, F_PROC),
          ("3×3 depthwise", "one filter per channel\nReLU6", 1.6, F_MODEL),
          ("1×1 conv", "project back\nlinear (no ReLU)", 0.55, F_PROC)]
xb = 0.8
pos = []
for t, sub, hh, st in blocks:
    node(s, xb, y_c - (0.7 + hh * 0.9) / 2, 1.95, 0.7 + hh * 0.9, t, sub, size=14, sub_size=11.5, **st)
    pos.append(xb)
    xb += 1.95 + 0.45
for i in range(1, len(pos)):
    arrow(s, pos[i] - 0.43, y_c, pos[i] - 0.03, y_c, color=MUTED, width=2)
xa = xb
add = box(s, xa - 0.1, y_c - 0.3, 0.6, 0.6, fill=TERRA, shape=MSO_SHAPE.OVAL)
text(s, xa - 0.1, y_c - 0.3, 0.6, 0.6, "+", size=24, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
     anchor=MSO_ANCHOR.MIDDLE)
arrow(s, xa - 0.43, y_c, xa - 0.12, y_c, color=MUTED, width=2)
node(s, xa + 0.85, y_c - 0.6, 1.6, 1.2, "Output", "h × w × c", size=14, sub_size=11.5, **F_OUT)
arrow(s, xa + 0.52, y_c, xa + 0.83, y_c, color=MUTED, width=2)
# skip connection
sk_y = y_c - 1.65
line(s, pos[0] + 0.97, y_c - 0.55, pos[0] + 0.97, sk_y, color=TERRA, width=2)
line(s, pos[0] + 0.97, sk_y, xa + 0.2, sk_y, color=TERRA, width=2)
arrow(s, xa + 0.2, sk_y, xa + 0.2, y_c - 0.32, color=TERRA, width=2)
text(s, pos[1], sk_y - 0.38, 5, 0.35, "skip connection (when input and output shapes match)", size=12.5,
     color=TERRA, italic=True)
text(s, 0.8, 4.65, 7, 0.35, "narrow  →  wide  →  narrow: the reverse of a ResNet bottleneck, hence “inverted”",
     size=13, color=MUTED, italic=True)
pts = [("Depthwise-separable", "a normal 3×3 conv mixes space and channels at once: k²·C_in·C_out multiplies per "
                               "pixel. Splitting it into 3×3 per channel + 1×1 mix costs k²·C_in + C_in·C_out, about "
                               "8–9× fewer for k = 3."),
       ("Linear bottleneck", "no ReLU after the narrow projection, so information is not thrown away in the low-"
                             "dimensional layer."),
       ("Residual", "adding the input back keeps gradients flowing through 17 blocks.")]
for i, (h_, d) in enumerate(pts):
    x = 0.6 + i * 4.1
    box(s, x, 5.15, 3.9, 1.75, fill=SAND)
    text(s, x + 0.25, 5.25, 3.45, 0.4, h_, size=15, bold=True, color=NAVY)
    text(s, x + 0.25, 5.65, 3.5, 1.2, d, size=12, color=INK)

# ================================================================== why MobileNetV2
s = new_slide("Why MobileNetV2, and how it helped us", tag=M2, tag_color=TERRA, notes=(
    "Why this model and not a bigger one? The table compares it with two classic networks. MobileNetV2 has about 3.5 "
    "million parameters and needs about 300 million multiply-adds per image. ResNet-50 has about 25 million "
    "parameters and 4 billion multiply-adds, VGG-16 138 million and 15 billion. For us that mattered in four "
    f"concrete ways. It trained in about {train_min_txt} on a laptop CPU, because we had no GPU available. It runs "
    "inside a web browser on a phone in a fraction of a second, so the website needs no AI server. Its ImageNet "
    "features, edges and textures, transfer well to leaf lesions, so a few thousand photos were enough. And because "
    "it ends in global average pooling and a single linear layer, the heatmap we draw is exact rather than an "
    "approximation."))
tbl = [("", "Parameters", "Multiply-adds / image", "Designed for"),
       ("MobileNetV2 (ours)", "~3.5 M", "~0.3 B", "phones"),
       ("ResNet-50", "~25.6 M", "~4.1 B", "servers / GPUs"),
       ("VGG-16", "~138 M", "~15.5 B", "servers / GPUs")]
for r, row in enumerate(tbl):
    y = 1.55 + r * 0.55
    if r:
        box(s, 0.6, y - 0.05, 12.1, 0.48, fill=SLATE_TINT if r == 1 else (SAND if r % 2 == 0 else WHITE), radius=0.15)
    for c, v in enumerate(row):
        xs = [0.85, 4.4, 7.3, 10.3][c]
        text(s, xs, y, 2.9 if c == 0 else 2.6, 0.4, v, size=15, bold=(r == 0 or c == 0 or r == 1),
             color=NAVY if (r == 0 or c == 0) else INK)
text(s, 0.6, 3.85, 12, 0.45, "How it helped this project", size=20, bold=True, color=NAVY, font=HEAD)
helps = [("Trains on a laptop", f"no GPU available: whole training took about {train_min_txt} on the CPU"),
         ("Runs in the browser", "small and fast enough to run on a phone with ONNX Runtime Web, so no AI server"),
         ("Transfers to leaves", "ImageNet edge and texture detectors fit lesions; a few thousand photos suffice"),
         ("Exact heatmap", "global average pool + one linear layer makes the class activation map exact")]
for i, (h_, d) in enumerate(helps):
    x = 0.6 + i * 3.1
    box(s, x, 4.45, 2.9, 2.3)
    badge(s, x + 0.25, 4.65, str(i + 1), fill=TERRA, d=0.5, size=14)
    text(s, x + 0.9, 4.66, 1.95, 0.6, h_, size=15, bold=True, color=NAVY)
    text(s, x + 0.25, 5.35, 2.5, 1.35, d, size=13, color=MUTED)

# ================================================================== training
s = new_slide("Training: transfer learning in two phases", tag=M2, tag_color=TERRA, notes=(
    f"Training happens in two phases. In phase one the whole pretrained body is frozen and only the new "
    f"{n_classes}-class linear layer is trained, so we don't damage what the network already knows. In phase two we "
    f"unfreeze the last {ft_blocks} feature blocks and fine-tune them together with the classifier at a small "
    "learning rate, so the high-level features adapt to leaves. We keep the epoch with the best validation accuracy "
    "and report the score once, on the untouched test set. The chart shows validation accuracy over the epochs: "
    "the head alone gets most of the way, and fine-tuning adds the rest. The model is then exported to ONNX for the "
    "browser."))
steps_ = [("Phase 1: train the head", f"body frozen, new {n_classes}-class layer"
           + (f" · {ep_head} epochs" if ep_head else ""), F_PROC),
          ("Phase 2: fine-tune", f"last {ft_blocks} blocks + head, small learning rate"
           + (f" · {ep_ft} epochs" if ep_ft else ""), F_MODEL),
          ("Pick and test", "best epoch on validation; score once on the test set", F_PROC),
          ("Export", "ONNX for ONNX Runtime Web", F_OUT)]
for i, (t, sub, st) in enumerate(steps_):
    x = 0.6 + i * 3.15
    node(s, x, 1.5, 2.8, 1.45, t, sub, size=15, sub_size=12.5, **st)
    if i:
        arrow(s, x - 0.33, 2.22, x - 0.04, 2.22, color=SLATE)
facts = [("Input", f"{input_size}×{input_size} RGB, ImageNet normalisation"),
         ("Augmentation", "random crops, flips, colour jitter"),
         ("Optimiser", "Adam, cross-entropy loss"),
         ("Training time", f"about {train_min_txt} on a laptop CPU")]
for i, (k, v) in enumerate(facts):
    y = 3.35 + i * 0.52
    text(s, 0.6, y, 2.0, 0.4, k, size=14.5, bold=True, color=NAVY)
    text(s, 2.55, y, 3.9, 0.4, v, size=14.5, color=INK)
box(s, 0.6, 5.55, 5.9, 1.3, fill=NAVY, radius=0.08)
text(s, 0.85, 5.55, 5.45, 1.3, [("Why two phases?", {"bold": True, "color": GOLD}),
                               ("Training the new head first protects the pretrained features; fine-tuning then adapts "
                                "them to leaf textures without forgetting what they already know.", {"color": WHITE})],
     size=13.5, anchor=MSO_ANCHOR.MIDDLE, spacing=2)
if (RES / "fig_leaf_training.png").exists():
    image(s, RES / "fig_leaf_training.png", 6.8, 3.25, w=6.0, h=2.3)
    explain(s, 6.8, 5.65, 6.0, "x = epoch, y = accuracy on the validation images; first the head phase, then "
            "fine-tuning.", "the frozen-feature head already does most of the work; fine-tuning the later blocks "
            "adds the rest.", size=12, h=1.25)

# ================================================================== leaf results
s = new_slide("Result: " + (f"{acc_txt} accuracy on unseen leaf images" if acc_txt != "—"
                            else "how well it diagnoses unseen leaf images"), tag=M2, tag_color=TERRA, notes=(
    f"On the test set, images the model never saw in training, it reaches {acc_txt} accuracy and a macro F1 of "
    f"{f1_txt}. Macro F1 averages over all {n_classes} classes equally, so a weak class cannot hide behind the "
    "others. The confusion matrix: rows are the true disease, columns the prediction, and the diagonal is correct. "
    "Almost everything sits on the diagonal. The few mistakes are between diseases that look alike, mostly the "
    f"tomato spot diseases; the weakest classes are {weak_txt}."))
stat(s, 0.6, 1.45, 3.6, acc_txt, "test accuracy on unseen images", value_color=TERRA, size=50)
stat(s, 0.6, 3.05, 3.6, f1_txt, "macro F1 (all classes weighted equally)", value_color=NAVY, size=50)
stat(s, 0.6, 4.65, 3.6, fmt_int(n_test_img), "test images, never seen in training", value_color=NAVY, size=50)
picture(s, RES / "fig_leaf_confusion.png", 4.5, 1.35, 4.9, 4.2, "Confusion matrix on the test set")
explain(s, 9.6, 1.45, 3.2, "rows = true disease, columns = predicted; the diagonal is correct, anything off it "
        "is a mistake.", f"almost all on the diagonal; the confusions are among look-alike tomato spot diseases "
        f"({weak_txt}).", size=12, h=3.3)
text(s, 4.5, 5.75, 8.3, 1.1, [[("Weakest classes:  ", {"bold": True, "color": NAVY}),
                               (", ".join(f"{c.get('crop')} {c.get('name').lower()} (F1 {c.get('f1', 0):.2f})"
                                          for c in weakest) or "—", {"color": INK})]], size=13)

# ================================================================== heatmap
s = new_slide("Explainable: the heatmap shows where the model looked", tag=M2, tag_color=TERRA, notes=(
    "A diagnosis is more trustworthy if we can see why. For every prediction we draw a class activation map: the "
    "classifier's weights for the predicted class multiplied with the network's last 7 by 7 feature maps, and "
    "summed. Because the class score is exactly the average of this map plus a bias, the map is not a guess. In the "
    "figure, the top row is test photos with the true and predicted label, and the bottom row the same photos with "
    "the heatmap: red is the strongest evidence. The heat sits on the lesions, not on the background, so the model "
    "is using the right evidence. One caveat: PlantVillage photos are lab images on plain backgrounds, so field "
    "photos will be harder."))
picture(s, RES / "fig_leaf_cam_examples.png", 0.55, 1.4, 7.6, 4.15, "Leaf photos with class-activation heatmaps")
explain(s, 0.6, 5.7, 7.5, "top: test photos with true / predicted label; bottom: same photos with the heatmap "
        "(red = strongest evidence).", "the heat sits on lesions and discoloured patches, not the background: the "
        "model uses the right evidence.", size=12, h=1.15)
pts = [("How it is computed", "CAM_c(x, y) = Σ_k w_c,k · f_k(x, y): classifier weights × last feature maps"),
       ("Why it is exact", "class score = average of CAM_c + bias, because of global average pooling"),
       ("Why it matters", "an expert can check the model looks at the disease, not the background")]
for i, (h_, d) in enumerate(pts):
    y = 1.5 + i * 1.25
    badge(s, 8.55, y, str(i + 1), fill=TERRA, d=0.5, size=14)
    text(s, 9.2, y - 0.02, 3.6, 0.4, h_, size=15.5, bold=True, color=NAVY)
    text(s, 9.2, y + 0.37, 3.6, 0.85, d, size=12.5, color=MUTED)
box(s, 8.55, 5.4, 4.25, 1.35, fill=TERRA_TINT)
text(s, 8.75, 5.45, 3.9, 1.25, [("Caveat", {"bold": True, "color": TERRA, "size": 14}),
                               ("PlantVillage photos are single leaves on plain backgrounds; field photos with soil, "
                                "other leaves and harsh light will be harder.", {"color": INK, "size": 12.5})],
     anchor=MSO_ANCHOR.MIDDLE, spacing=2)

# ================================================================== website architecture
s = new_slide("The website: static files, models run in the browser", tag=WEB, tag_color=MUTED, notes=(
    "Both modules are deployed as one static website on GitHub Pages. Static means the server only hands out "
    "files: the page and its JavaScript, the agent's Q-tables and all precomputed seasons as JSON, and the leaf "
    "model as an ONNX file. All the computing happens in the visitor's browser. The irrigation tab reads the JSON "
    "and draws any season, the rulebook and the results. The leaf doctor tab preprocesses the photo in JavaScript, "
    "runs the network with ONNX Runtime Web, which uses WebAssembly, and computes the softmax and the heatmap. We "
    "precomputed the irrigation seasons in Python so the website shows exactly the numbers in our report."))
box(s, 0.6, 1.6, 3.5, 4.5, fill=NAVY, radius=0.08)
text(s, 0.85, 1.8, 3.0, 0.5, "GitHub Pages", size=20, bold=True, color=WHITE, font=HEAD)
text(s, 0.85, 2.5, 3.0, 3.5, [("serves static files", {"color": GOLD, "bold": True}),
                              ("index.html + JavaScript", {"color": WHITE}),
                              ("Q-tables + 225 seasons (JSON)", {"color": WHITE}),
                              ("results + weather (JSON)", {"color": WHITE}),
                              ("leaf model (ONNX) + classes", {"color": WHITE})], size=14, spacing=8)
arrow(s, 4.2, 3.85, 5.05, 3.85, color=SLATE, width=3)
text(s, 4.05, 3.35, 1.2, 0.4, "HTTPS", size=12, color=MUTED, align=PP_ALIGN.CENTER)
box(s, 5.15, 1.6, 7.6, 4.5, fill=SAND, radius=0.05)
text(s, 5.4, 1.72, 7.1, 0.4, "Visitor's browser (phone or laptop)", size=17, bold=True, color=NAVY)
chain(s, [("Smart irrigation", "season JSON", F_PROC), ("Charts", "ECharts", F_PROC),
          ("Rulebook + results", "per crop / city", F_OUT)], 5.4, 2.35, 2.15, 1.2, 0.3, color=SLATE, size=14,
      sub_size=11.5)
chain(s, [("Leaf photo", "resize, crop, normalise", F_DATA), ("CNN", "ONNX Runtime Web", F_MODEL),
          ("Diagnosis", "top-3 + heatmap", F_OUT)], 5.4, 4.05, 2.15, 1.2, 0.3, color=TERRA, size=14, sub_size=11.5)
text(s, 5.4, 5.5, 7.1, 0.5, "Three tabs: Overview · Smart irrigation · Leaf doctor", size=13.5, color=MUTED,
     italic=True)
text(s, 0.6, 6.4, 12.1, 0.5, "Live at  " + SITE_URL, size=18, bold=True, color=SLATE)

# ================================================================== website tour
s = new_slide("A quick tour of the website", tag=WEB, tag_color=MUTED, notes=(
    "A quick tour before the live demo. The overview introduces both modules. The smart irrigation tab lets you "
    "pick a crop, city, season and year and compare all four policies day by day, and it also shows the learned "
    "rulebook and the test results. The leaf doctor tab takes a photo and returns the top three diagnoses with "
    "confidence, a heatmap and advice."))
shots = [("home.png", "Overview: both modules"), ("irrigation.png", "Smart irrigation: season simulator"),
         ("leaf.png", "Leaf doctor: diagnosis + heatmap"), ("agent.png", "Smart irrigation: the learned rulebook")]
for i, (fn, cap) in enumerate(shots):
    x = 0.6 + (i % 2) * 6.15
    y = 1.35 + (i // 2) * 2.9
    picture(s, SCREENS / fn, x, y, 5.95, 2.45, cap, frame=True, fill=True)
    text(s, x, y + 2.48, 5.95, 0.35, cap, size=12.5, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

# ================================================================== live demo
s = new_slide(dark=True, notes=(
    "Now the live demo. Open the link or scan the QR code. First, smart irrigation: pick Jodhpur, maize, 2023 and "
    "show the agent's soil line staying above the stress line with less water. Second, the rulebook: switch the "
    "forecast and point out that the agent waters less when rain is coming. Third, the leaf doctor: load a sample "
    "tomato leaf and show the top-3 diagnosis and the heatmap on the lesions."))
text(s, 0.9, 0.8, 8, 0.5, "LIVE DEMO", size=16, bold=True, color=GOLD)
text(s, 0.9, 1.3, 8.5, 1.0, "Try it on your phone", size=44, bold=True, font=HEAD, color=WHITE)
text(s, 0.9, 2.5, 8.6, 0.8, SITE_URL.replace("https://", ""), size=30, bold=True, color=GOLD)
for i, d in enumerate(["Smart irrigation: Jodhpur, maize, 2023: agent vs sensor rule",
                       "Rulebook: watch the agent hold back when rain is forecast",
                       "Leaf doctor: diagnose a sample tomato leaf, see the heatmap"]):
    badge(s, 0.9, 3.75 + i * 0.85, str(i + 1), fill=SLATE if i < 2 else TERRA, d=0.5, size=14)
    text(s, 1.6, 3.82 + i * 0.85, 7.6, 0.5, d, size=17, color=WHITE)
qr_path = Path(tempfile.gettempdir()) / "agrisense_qr.png"
try:
    import cv2
    import numpy as np
    qr = cv2.QRCodeEncoder.create().encode(SITE_URL)
    qr = cv2.copyMakeBorder(qr, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
    qr = cv2.resize(qr, (qr.shape[1] * 16, qr.shape[0] * 16), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(qr_path), qr.astype(np.uint8))
    box(s, 9.75, 1.45, 3.0, 3.0, fill=WHITE, radius=0.06)
    image(s, qr_path, 9.9, 1.6, w=2.7, h=2.7)
    text(s, 9.75, 4.6, 3.0, 0.4, "scan to open", size=14, color=GOLD, align=PP_ALIGN.CENTER)
except Exception:  # QR is a nice-to-have; never fail the build over it
    pass

# ================================================================== limitations
s = new_slide("Limitations and next steps", notes=(
    "Our limitations. For irrigation: it is a simulated field, not a field trial; the yield model does not know "
    "that stress at flowering hurts more; the forecast is synthetic; and the coarse table is why Ludhiana winters "
    "are weaker. For leaf disease: PlantVillage photos are lab images on plain backgrounds, it covers only maize "
    "and tomato, and the heatmap is coarse. Next steps: a deep Q-network on continuous inputs with real soil-moisture "
    "sensors and forecasts; fine-tuning the CNN on field photos and more crops; and linking the two modules, so a "
    "detected disease adjusts the irrigation plan."))
lim = [("Irrigation", "Simulated field (FAO-56), not a field trial; seasonal yield model; synthetic forecast; "
                      "coarse 360-state table (weaker in Ludhiana winters)"),
       ("Leaf disease", "PlantVillage = lab photos on plain backgrounds; maize + tomato only; coarse 7×7 heatmap; "
                        "no severity estimate")]
nxt = [("Irrigation", "Deep Q-network on continuous inputs; real forecasts and IoT soil-moisture sensors; soil type "
                      "as an input"),
       ("Leaf disease", "Fine-tune the whole network on field photos (e.g. PlantDoc) with a GPU; more crops; severity"),
       ("Together", "Link the modules: a detected disease adjusts the irrigation plan")]
for i, (title, items, col, mark) in enumerate([("Limitations", lim, TERRA, "!"), ("Next steps", nxt, SLATE, "→")]):
    x = 0.6 + i * 6.2
    box(s, x, 1.55, 5.9, 5.25)
    badge(s, x + 0.3, 1.8, mark, fill=col, d=0.6, size=18)
    text(s, x + 1.1, 1.85, 4.5, 0.5, title, size=22, bold=True, color=NAVY, font=HEAD)
    for j, (h_, d) in enumerate(items):
        y = 2.7 + j * 1.35
        text(s, x + 0.4, y, 5.2, 0.35, h_, size=15, bold=True, color=col)
        text(s, x + 0.4, y + 0.35, 5.2, 0.95, d, size=14, color=INK)

# ================================================================== conclusion
s = new_slide(dark=True, notes=(
    f"To conclude. Module one: a simple, explainable Q-learning agent used {save_fix:.0f} percent less water than "
    "calendar irrigation with a higher yield, and matched the FAO expert rule while using "
    f"{save_thr:.1f} percent less water. Module two: a MobileNetV2 network diagnoses {n_classes} maize and tomato "
    f"leaf conditions with {acc_txt} accuracy on unseen images and shows where it looked. Both run on the website "
    "on screen. Thank you, we're happy to take questions."))
text(s, 0.9, 0.8, 11, 0.9, "Conclusion", size=40, bold=True, font=HEAD, color=WHITE)
pts = [(f"{save_fix:.0f} %", "less irrigation water than a fixed calendar, with a higher yield"),
       (f"{save_thr:.1f} %", "less water than the FAO expert rule, learned without being told it"),
       (f"{s_rl.avg_relative_yield_pct:.1f} %", "of maximum yield kept by the RL agent on unseen seasons"),
       (acc_txt, f"leaf-disease accuracy on unseen images ({n_classes} classes)")]
for i, (v, l) in enumerate(pts):
    x = 0.9 + i * 3.05
    text(s, x, 2.0, 2.9, 1.0, v, size=44, bold=True, font=HEAD, color=GOLD)
    text(s, x, 3.05, 2.75, 1.0, l, size=15, color=WHITE)
text(s, 0.9, 5.0, 11.5, 0.9, "Thank you. Questions?", size=36, bold=True, font=HEAD, color=WHITE)
text(s, 0.9, 5.95, 11.5, 0.5, SITE_URL, size=18, color=GOLD, bold=True)

prs.save(OUT)
print("saved", OUT, len(prs.slides), "slides")
