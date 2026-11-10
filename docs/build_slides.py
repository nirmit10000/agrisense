"""Builds docs/AgriSense_Presentation.pptx from the latest results.   python docs/build_slides.py

Story: problem -> need -> solution & architecture -> Module 1 (RL irrigation) -> Module 2 (leaf disease CNN)
-> website -> live demo -> limitations -> conclusion.

Every slide has speaker notes (View > Notes in PowerPoint) that you can read while presenting.
Inputs that do not exist yet (leaf metrics/figures, website screenshots) are drawn as labelled
placeholders, so re-running this script after they appear fills them in.
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
from agrisense.config import CITIES, IRRIGATION_ACTIONS_MM, RewardConfig, TrainConfig  # noqa: E402

RES, DOCS = ROOT / "results", ROOT / "docs"
SCREENS = DOCS / "screens"
OUT = DOCS / "AgriSense_Presentation.pptx"
SITE_URL = "https://nirmit10000.github.io/agrisense/"

# palette: deep field green dominates, leaf green supports, water blue is the accent
GREEN_DK = RGBColor(0x17, 0x3F, 0x2A)
GREEN = RGBColor(0x2F, 0x6B, 0x3F)
LEAF = RGBColor(0x8C, 0xC0, 0x6A)
TINT = RGBColor(0xEE, 0xF4, 0xEC)
BLUE_TINT = RGBColor(0xE6, 0xF0, 0xFB)
WATER = RGBColor(0x2A, 0x78, 0xD6)
RUST = RGBColor(0xC0, 0x5A, 0x1E)
INK = RGBColor(0x1A, 0x1A, 0x1A)
INK2 = RGBColor(0x55, 0x5B, 0x57)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PH_FILL, PH_LINE = RGBColor(0xF6, 0xF7, 0xF5), RGBColor(0xC9, 0xD3, 0xC9)
POLICY = {"Q-learning agent (RL)": RGBColor(0x2A, 0x78, 0xD6), "Sensor threshold rule": RGBColor(0xEB, 0x68, 0x34),
          "Fixed schedule": RGBColor(0x1B, 0xAF, 0x7A), "Rainfed (no irrigation)": RGBColor(0xED, 0xA1, 0x00)}
HEAD, BODY = "Cambria", "Calibri"

# ------------------------------------------------------------------ irrigation results
summary = pd.read_csv(RES / "summary_by_policy.csv", index_col=0)
evals = pd.read_csv(RES / "eval_results.csv")
RL, THR, FIX, RAIN = summary.index.tolist()
s_rl, s_thr, s_fix = summary.loc[RL], summary.loc[THR], summary.loc[FIX]
n_test = len(evals) // 4
save_fix = 100 * (1 - s_rl.avg_irrigation_mm / s_fix.avg_irrigation_mm)
save_thr = 100 * (1 - s_rl.avg_irrigation_mm / s_thr.avg_irrigation_mm)
rw, tc = RewardConfig(), TrainConfig()

# the example season shown on the "one season" slide (same as results/fig_season_trace.png)
from agrisense import weather as wx  # noqa: E402
from agrisense.agents import ThresholdPolicy  # noqa: E402
from agrisense.config import CROPS, DEFAULT_SOIL  # noqa: E402
from agrisense.env import IrrigationEnv  # noqa: E402
from agrisense.simulate import all_policies, run_season  # noqa: E402
_trace = {}
for _pol in [all_policies("Maize")[0], ThresholdPolicy()]:
    _env = IrrigationEnv(wx.load("Jodhpur"), CROPS["Maize"], DEFAULT_SOIL, seed=123 + 2023)
    _trace[_pol.name] = run_season(_env, _pol, _env.index_of("2023-06-15"))[0]
trace_rl, trace_thr = _trace[RL]["irrigation_mm"], _trace[THR]["irrigation_mm"]

# ------------------------------------------------------------------ leaf results (may not exist yet)
_leaf_path = RES / "leaf_metrics.json"
leaf = json.loads(_leaf_path.read_text(encoding="utf-8")) if _leaf_path.exists() else {}
DEFAULT_CLASSES = (
    [("Maize", n) for n in ["Cercospora (gray) leaf spot", "Common rust", "Northern leaf blight", "Healthy"]]
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
size_mb = leaf.get("model_size_mb")
size_txt = f"{size_mb:.1f} MB" if isinstance(size_mb, (int, float)) else "—"
_ep = leaf.get("epochs", {}) or {}
train_min = leaf.get("train_minutes")
input_size = leaf.get("input_size", 224)
_pc = leaf.get("per_class") or []
worst = min(_pc, key=lambda c: c.get("f1", 1)) if _pc else None


def fmt_int(v):
    return f"{int(v):,}" if isinstance(v, (int, float)) else "—"


prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = 13.333, 7.5


# ------------------------------------------------------------------ helpers
def text(slide, x, y, w, h, content, size=16, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, italic=False, spacing=None):
    """content: str or list of str/(str, dict) paragraphs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = content if isinstance(content, list) else [content]
    for i, ptxt in enumerate(paras):
        opts = {}
        if isinstance(ptxt, tuple):
            ptxt, opts = ptxt
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = opts.get("align", align)
        if spacing:
            p.space_after = Pt(spacing)
        r = p.add_run()
        r.text = ptxt
        f = r.font
        f.name = opts.get("font", font)
        f.size = Pt(opts.get("size", size))
        f.bold = opts.get("bold", bold)
        f.italic = opts.get("italic", italic)
        f.color.rgb = opts.get("color", color)
    return tb


def box(slide, x, y, w, h, fill=TINT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line=None, radius=0.08):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line:
        s.line.color.rgb = line
        s.line.width = Pt(1.25)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    return s


def badge(slide, x, y, label, fill=GREEN, d=0.55, size=16):
    """The deck's motif: a numbered/lettered circle."""
    c = box(slide, x, y, d, d, fill=fill, shape=MSO_SHAPE.OVAL)
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), True, WHITE, BODY
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
    """Image if the file exists, otherwise a tidy labelled placeholder of the same size.

    fill=True fills the whole box and crops the bottom of a tall image (for website screenshots)."""
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
            pic.line.color.rgb = RGBColor(0xD0, 0xD5, 0xD0)
        return pic
    box(slide, x, y, w, h, fill=PH_FILL, line=PH_LINE, radius=0.04)
    text(slide, x + 0.2, y, w - 0.4, h, [label, (f"placeholder: {Path(path).name}", {"size": 11, "color": INK2})],
         size=15, color=GREEN, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    return None


def _arrowhead(connector):
    from lxml import etree
    from pptx.oxml.ns import qn
    ln = connector.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("h", "med")


def line(slide, x1, y1, x2, y2, color=INK2, width=2.0, head=False, dash=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    if dash:
        c.line.dash_style = 4
    if head:
        _arrowhead(c)
    return c


def arrow(slide, x1, y1, x2, y2, color=INK2, width=2.25):
    return line(slide, x1, y1, x2, y2, color=color, width=width, head=True)


def node(slide, x, y, w, h, title, sub=None, fill=TINT, title_color=GREEN_DK, sub_color=INK2, size=14,
         sub_size=11.5, line_color=None):
    """Flowchart node: rounded box with a bold title and an optional smaller line."""
    box(slide, x, y, w, h, fill=fill, line=line_color, radius=0.12)
    paras = [(title, {"bold": True, "color": title_color, "size": size, "align": PP_ALIGN.CENTER})]
    if sub:
        paras.append((sub, {"color": sub_color, "size": sub_size, "align": PP_ALIGN.CENTER}))
    text(slide, x + 0.1, y + 0.06, w - 0.2, h - 0.12, paras, anchor=MSO_ANCHOR.MIDDLE, spacing=2)


def chain(slide, nodes, x0, y, w, h, gap, color=INK2, **kw):
    """A horizontal row of nodes joined by arrows. nodes: list of (title, sub, fill)."""
    for i, (title, sub, fill) in enumerate(nodes):
        x = x0 + i * (w + gap)
        node(slide, x, y, w, h, title, sub, fill=fill, **kw)
        if i:
            arrow(slide, x - gap + 0.04, y + h / 2, x - 0.04, y + h / 2, color=color, width=2)


def new_slide(title=None, dark=False, notes="", tag=None, tag_color=None):
    s = prs.slides.add_slide(BLANK)
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = GREEN_DK if dark else WHITE
    if tag:
        text(s, 0.6, 0.2, 8, 0.3, tag, size=11, bold=True, color=tag_color or GREEN)
    if title:
        text(s, 0.6, 0.45, W - 1.2, 0.8, title, size=30, bold=True, font=HEAD, color=WHITE if dark else GREEN_DK)
    s.notes_slide.notes_text_frame.text = notes or "(no notes)"
    num = len(prs.slides)
    if num > 1:
        text(s, W - 1.0, H - 0.45, 0.5, 0.3, str(num), size=10, color=LEAF if dark else INK2, align=PP_ALIGN.RIGHT)
    return s


def stat(slide, x, y, w, value, label, value_color=GREEN_DK, label_color=INK2, size=54, label_size=14):
    text(slide, x, y, w, 1.0, value, size=size, bold=True, font=HEAD, color=value_color)
    text(slide, x, y + 1.05, w, 0.9, label, size=label_size, color=label_color)


M1 = "MODULE 1 · SMART IRRIGATION (REINFORCEMENT LEARNING)"
M2 = "MODULE 2 · LEAF DISEASE DETECTION (DEEP LEARNING)"
WEB = "DEPLOYMENT · THE WEBSITE"

# ================================================================== 1. title
s = new_slide(dark=True, notes=(
    "Good morning. Our project is AgriSense: AI for smart irrigation and crop health. It falls under the "
    "course theme of agriculture: intelligent irrigation, crop monitoring and resource optimisation. It has two "
    "AI modules: a reinforcement learning agent that decides when and how much to irrigate, and a deep learning "
    "model that recognises leaf diseases from a photo. Both run on a public website."))
text(s, 0.9, 1.35, 11, 0.4, "EC-422 ARTIFICIAL INTELLIGENCE  ·  COURSE PROJECT", size=14, color=LEAF, bold=True)
text(s, 0.9, 1.95, 11, 1.4, "AgriSense", size=72, bold=True, font=HEAD, color=WHITE)
text(s, 0.9, 3.3, 11.5, 0.8, "AI for smart irrigation and crop health", size=30, color=WHITE, font=HEAD)
for i, (lbl, col) in enumerate([("Module 1  ·  Reinforcement learning decides when and how much to irrigate", WATER),
                                ("Module 2  ·  A deep CNN diagnoses leaf diseases from a photo", LEAF)]):
    badge(s, 0.9, 4.3 + i * 0.6, str(i + 1), fill=col, d=0.42, size=13)
    text(s, 1.5, 4.36 + i * 0.6, 10, 0.4, lbl, size=17, color=WHITE)
text(s, 0.9, 5.95, 8, 0.8, ["[Student 1 name]  ·  [Roll no.]", "[Student 2 name]  ·  [Roll no.]"], size=16,
     color=WHITE)
text(s, 7.4, 6.2, 5.2, 0.4, SITE_URL.replace("https://", ""), size=16, color=LEAF, bold=True, align=PP_ALIGN.RIGHT)

# ================================================================== 2. problem statement
s = new_slide("Problem statement: two decisions every farmer makes", notes=(
    "The course theme we picked is agriculture: intelligent irrigation, crop monitoring and resource "
    "optimisation. On a farm this comes down to two decisions that are still mostly made by habit or guesswork. "
    "First, water: when should I irrigate, and how much? Second, crop health: this leaf looks wrong, is it a "
    "disease, and which one? Our problem statement is to support both decisions with AI, in a form any farmer "
    "could open on a phone."))
box(s, 0.6, 1.55, 12.1, 1.05, fill=GREEN_DK)
text(s, 0.95, 1.55, 11.5, 1.05, [("Course theme  ", {"bold": True, "color": LEAF, "size": 14}),
                                ("Agriculture: intelligent irrigation, crop monitoring and resource optimisation",
                                 {"color": WHITE, "size": 20, "font": HEAD})], anchor=MSO_ANCHOR.MIDDLE)
for i, (b, col, tint, q, d) in enumerate([
        ("1", WATER, BLUE_TINT, "Water: when, and how much, should I irrigate today?",
         "Decided by the calendar or by feel. Too much wastes water and drains away; too little stresses the crop."),
        ("2", GREEN, TINT, "Health: is this leaf diseased, and with what?",
         "Needs an expert who is rarely nearby. Late or wrong diagnosis means lost yield and wasted sprays.")]):
    x = 0.6 + i * 6.2
    box(s, x, 2.9, 5.9, 2.4, fill=tint)
    badge(s, x + 0.3, 3.15, b, fill=col, d=0.6, size=18)
    text(s, x + 1.1, 3.15, 4.6, 0.9, q, size=19, bold=True, color=GREEN_DK)
    text(s, x + 1.1, 4.1, 4.5, 1.1, d, size=14, color=INK2)
text(s, 0.6, 5.6, 12, 0.4, "Our problem statement", size=18, bold=True, color=GREEN_DK, font=HEAD)
text(s, 0.6, 6.0, 12.1, 0.9, "Use AI to recommend daily irrigation that saves water without losing yield, and to "
     "diagnose leaf diseases from a photo, delivered as a free website that works on a phone.", size=16, color=INK)

# ================================================================== 3. the need
s = new_slide("Why it matters: water is wasted and crops are lost", notes=(
    "Why does this matter? Agriculture takes about 90 percent of India's freshwater withdrawals, according to "
    "FAO AQUASTAT. In our own simulation of a typical fixed calendar, about 350 millimetres per season simply "
    "drains below the roots, because the calendar ignores rain, and the crop still loses yield in hot spells. "
    "On the crop-health side, the FAO estimates that up to 40 percent of global crop production is lost to "
    "plant pests and diseases every year. Early, correct diagnosis is the first step to preventing that."))
cards = [("~90 %", "of India's freshwater withdrawals go to agriculture", "FAO AQUASTAT", GREEN_DK),
         (f"{s_fix.avg_wasted_mm:.0f} mm", "of irrigation per season drains away unused under a fixed 7-day calendar",
          "our simulation, 60 test seasons", WATER),
         ("40 %", "Up to this share of global crop production is lost to plant pests and diseases each year",
          "FAO", RUST)]
for i, (v, l, src, col) in enumerate(cards):
    x = 0.6 + i * 4.1
    box(s, x, 1.6, 3.8, 3.3)
    text(s, x + 0.35, 1.85, 3.2, 1.0, v, size=50, bold=True, font=HEAD, color=col)
    text(s, x + 0.35, 2.95, 3.2, 1.2, l, size=16, color=INK)
    text(s, x + 0.35, 4.35, 3.2, 0.4, "Source: " + src, size=11, color=INK2, italic=True)
box(s, 0.6, 5.25, 12.1, 1.5, fill=GREEN_DK)
text(s, 0.95, 5.25, 11.4, 1.5, [("The gap", {"bold": True, "color": LEAF, "size": 14}),
                               ("Both decisions are made without data: irrigation by the calendar, diagnosis by eye. "
                                "Both are the kind of pattern-and-decision problems AI is good at.",
                                {"color": WHITE, "size": 18})], anchor=MSO_ANCHOR.MIDDLE, spacing=4)

# ================================================================== 4. our solution
s = new_slide("Our solution: AgriSense, two AI modules in one website", notes=(
    "Our solution, AgriSense, has two AI modules. Module one is a reinforcement learning agent: it looks at soil "
    "moisture, crop stage, the rain forecast and how hot the day is, and recommends an irrigation amount. Module "
    "two is a convolutional neural network: you give it a photo of a maize or tomato leaf, and it returns the "
    "most likely disease, its confidence, a heatmap of where it looked, and advice. Both are packaged in one "
    "website that anyone can open."))
mods = [("1", WATER, BLUE_TINT, "Smart irrigation", "Reinforcement learning (tabular Q-learning)",
         ["Soil dryness, crop stage", "3-day rain forecast, evaporative demand"], "Irrigate 0–40 mm today",
         f"{save_fix:.0f} % less water than a fixed calendar"),
        ("2", GREEN, TINT, "Leaf disease detection", "Deep learning (MobileNetV2 CNN, transfer learning)",
         ["Photo of a maize or tomato leaf"], f"Top-3 diagnoses of {n_classes} classes + heatmap + advice",
         f"{acc_txt} accuracy on unseen test images")]
for i, (b, col, tint, name, method, ins, out, res) in enumerate(mods):
    x = 0.6 + i * 6.2
    box(s, x, 1.55, 5.9, 5.2, fill=tint)
    badge(s, x + 0.3, 1.8, b, fill=col, d=0.65, size=20)
    text(s, x + 1.15, 1.78, 4.6, 0.45, name, size=22, bold=True, color=GREEN_DK, font=HEAD)
    text(s, x + 1.15, 2.25, 4.6, 0.4, method, size=13, color=col, bold=True)
    for j, (lbl, body) in enumerate([("Input", ins), ("AI", [method.split(" (")[0]]), ("Output", [out])]):
        y = 2.95 + j * 0.95
        text(s, x + 0.35, y, 1.0, 0.4, lbl, size=13, bold=True, color=INK2)
        node(s, x + 1.35, y - 0.08, 4.25, 0.72, body[0], body[1] if len(body) > 1 else None, fill=WHITE,
             size=13, sub_size=12)
        if j < 2:
            arrow(s, x + 3.47, y + 0.66, x + 3.47, y + 0.85, color=col, width=1.75)
    text(s, x + 0.35, 5.95, 5.3, 0.6, res, size=17, bold=True, color=col)
text(s, 0.6, 6.85, 12.1, 0.4, "Both modules run inside one free website: " + SITE_URL, size=13, color=INK2)

# ================================================================== 5. architecture
s = new_slide("How it fits together: train offline, run in the browser", notes=(
    "Here is the whole system. On the left are the data sources: real weather from Open-Meteo, the FAO crop "
    "tables, and the PlantVillage leaf images. In the middle, we train both models offline in Python on a laptop. "
    "The trained models are then exported as plain files: the agent's Q-tables and precomputed seasons as JSON, "
    "and the neural network as an ONNX file. Those files are served by GitHub Pages as a static website. There "
    "is no server doing AI: the visitor's browser loads the files and runs both models itself, which is free, "
    "fast, and means a leaf photo never leaves the phone."))
cols = [("1", "Data", 0.55), ("2", "Train offline (Python)", 3.65), ("3", "Export as files", 6.75),
        ("4", "Serve and run", 9.85)]
for b, lbl, x in cols:
    badge(s, x, 1.45, b, fill=GREEN, d=0.45, size=13)
    text(s, x + 0.55, 1.5, 2.6, 0.4, lbl, size=15, bold=True, color=GREEN_DK)
cw, rows = 2.7, [(2.15, BLUE_TINT, WATER), (4.2, TINT, GREEN)]
row_nodes = [
    [("Weather + crop data", "Open-Meteo 2010-2025, 3 cities; FAO-56 crop tables"),
     ("Q-learning agent", "50,000 simulated seasons per crop, 3 crops"),
     ("Q-tables + seasons", "JSON: learned rules and precomputed simulations")],
    [("Leaf images", f"PlantVillage, maize + tomato, {n_classes} classes"),
     ("MobileNetV2 CNN", "ImageNet-pretrained, fine-tuned (transfer learning)"),
     ("Leaf model", f"ONNX file ({size_txt}) + class list")],
]
for (y, fill, col), nodes in zip(rows, row_nodes):
    for i, (t, sub) in enumerate(nodes):
        x = cols[i][2]
        node(s, x, y, cw, 1.55, t, sub, fill=fill, size=15, sub_size=12)
        if i:
            arrow(s, x - 0.37, y + 0.78, x - 0.04, y + 0.78, color=col)
    arrow(s, cols[2][2] + cw + 0.04, y + 0.78, cols[3][2] - 0.04, y + 0.78, color=col)
box(s, cols[3][2], 2.15, 2.95, 3.6, fill=GREEN_DK, radius=0.06)
text(s, cols[3][2] + 0.2, 2.25, 2.55, 3.4, [
    ("Static website", {"bold": True, "size": 18, "color": WHITE, "font": HEAD}),
    ("GitHub Pages", {"size": 13, "color": LEAF, "bold": True}),
    ("Irrigation simulator", {"size": 13, "color": WHITE}),
    ("Agent rulebook", {"size": 13, "color": WHITE}),
    ("Leaf Doctor (ONNX Runtime Web)", {"size": 13, "color": WHITE}),
    ("Results", {"size": 13, "color": WHITE}),
], anchor=MSO_ANCHOR.MIDDLE, spacing=5)
box(s, 0.55, 6.05, 12.25, 0.75, fill=TINT)
text(s, 0.85, 6.05, 11.8, 0.75, [("No AI server.  ", {"bold": True, "color": GREEN_DK}),
                                ("The trained models are just files; the visitor's browser does the computing, so it "
                                 "is free to host and a leaf photo never leaves the device.", {"color": INK})],
     size=15, anchor=MSO_ANCHOR.MIDDLE)

# ================================================================== 6. module 1 pipeline + why RL
s = new_slide("Module 1 · Smart irrigation with reinforcement learning", tag=M1, tag_color=WATER, notes=(
    "Module one. Each day the agent reads the state of the field: soil dryness, crop growth stage, the rain "
    "forecast and evaporative demand. It looks up the best action in its learned Q-table and recommends an "
    "irrigation dose. During training, a field simulator works out what that dose does to the soil and crop and "
    "returns a reward, which updates the table. Why reinforcement learning? Because irrigation is a sequence of "
    "decisions: today's watering changes tomorrow's soil, the yield is only known at the end of the season, "
    "rain is uncertain, and there is no labelled dataset of 'correct' irrigation, so the agent has to learn "
    "from consequences."))
nodes = [("Weather + forecast", "Open-Meteo, real data", BLUE_TINT),
         ("Field state", "soil dryness, crop stage, rain forecast, ET0", BLUE_TINT),
         ("Q-table lookup", "360 states × 5 actions", WATER),
         ("Irrigation dose", "0, 10, 20, 30 or 40 mm", BLUE_TINT),
         ("Field simulator", "FAO-56 water balance", TINT)]
nw, gap, y0 = 2.2, 0.32, 1.65
chain(s, [(t, sub, f) for t, sub, f in nodes], 0.6, y0, nw, 1.35, gap, color=WATER, size=15, sub_size=12)
# make the Q-table node text white (dark fill)
qx = 0.6 + 2 * (nw + gap)
node(s, qx, y0, nw, 1.35, "Q-table lookup", "360 states × 5 actions", fill=WATER, title_color=WHITE,
     sub_color=WHITE, size=15, sub_size=12)
sx = 0.6 + 4 * (nw + gap)
yb = y0 + 1.35 + 0.45
line(s, sx + nw / 2, y0 + 1.35, sx + nw / 2, yb, color=INK2, dash=True)
line(s, sx + nw / 2, yb, qx + nw / 2, yb, color=INK2, dash=True)
arrow(s, qx + nw / 2, yb, qx + nw / 2, y0 + 1.37, color=INK2, width=2)
text(s, qx + nw / 2 + 0.3, yb + 0.05, 5.5, 0.35, "reward updates the table (training only)", size=12, color=INK2,
     italic=True)
text(s, 0.6, 4.05, 12, 0.45, "Why reinforcement learning?", size=20, bold=True, color=GREEN_DK, font=HEAD)
reasons = [("Sequential", "today's watering changes tomorrow's soil moisture"),
           ("Delayed outcome", "yield is known only at the end of the season"),
           ("Uncertain", "rain forecasts are imperfect"),
           ("No labels", "no dataset of 'correct' irrigation, so it learns from consequences")]
for i, (h_, d) in enumerate(reasons):
    x = 0.6 + i * 3.1
    box(s, x, 4.65, 2.9, 2.05)
    badge(s, x + 0.25, 4.85, str(i + 1), fill=WATER, d=0.5, size=14)
    text(s, x + 0.9, 4.9, 1.9, 0.45, h_, size=16, bold=True, color=GREEN_DK)
    text(s, x + 0.25, 5.55, 2.5, 1.1, d, size=14, color=INK2)

# ================================================================== 7. data
s = new_slide("Data: 16 years of daily weather, three contrasting climates", tag=M1, tag_color=WATER, notes=(
    "We used the free Open-Meteo historical weather API, based on the ERA5 reanalysis. For every day from 2010 to "
    "2025 we have rainfall and reference evapotranspiration, ET0, the drying power of the air. We chose three very "
    "different climates so the agent can't just memorise one place: Pune with a strong monsoon, Jodhpur in the "
    "Thar desert, and Ludhiana in Punjab with cold winters. Seasons starting 2010 to 2020 are used for training; "
    "2021 to 2024 are kept aside for testing. The whole dataset is about half a megabyte."))
image(s, RES / "fig_climate.png", 0.5, 1.45, w=7.6, h=3.5)
for i, c in enumerate(CITIES):
    w_ = pd.read_csv(ROOT / "data" / f"weather_{c.lower()}.csv", parse_dates=["date"])
    w_ = w_[w_.date.dt.year < 2025]
    rain = w_.groupby(w_.date.dt.year).rain_mm.sum().mean()
    y = 1.55 + i * 1.15
    box(s, 8.5, y, 4.3, 1.0)
    text(s, 8.75, y + 0.12, 2.0, 0.4, c, size=18, bold=True, color=GREEN_DK)
    text(s, 10.8, y + 0.12, 1.85, 0.4, f"{rain:.0f} mm/yr", size=16, bold=True, color=WATER, align=PP_ALIGN.RIGHT)
    text(s, 8.75, y + 0.5, 3.9, 0.4, CITIES[c]["climate"].split(";")[0], size=13, color=INK2)
facts = [("Source", "Open-Meteo API (ERA5), free, no key"), ("Variables", "daily rain, ET0, Tmax, Tmin"),
         ("Train", "seasons starting 2010-2020"), ("Test", "seasons starting 2021-2024 (never seen)")]
for i, (k, v) in enumerate(facts):
    x = 0.6 + (i % 2) * 6.1
    y = 5.35 + (i // 2) * 0.65
    text(s, x, y, 1.4, 0.4, k, size=15, bold=True, color=GREEN_DK)
    text(s, x + 1.4, y, 4.6, 0.4, v, size=15, color=INK)

# ================================================================== 8. simulator
s = new_slide("The environment: a field simulator built on FAO-56", tag=M1, tag_color=WATER, notes=(
    "We can't run 50 thousand seasons on a real farm, so the agent trains in a simulator built on FAO-56, the "
    "international standard for crop water requirements. Think of the root zone as a bucket. Rain and irrigation "
    "fill it; anything above the brim drains away and is wasted. The crop drinks from it every day: ET0 times a "
    "crop coefficient Kc that changes with growth stage. While the soil holds more than a safe amount, called RAW, "
    "the crop is fine; below that line it is water-stressed. At the end of the season the FAO-33 equation turns "
    "the total water shortfall into a yield loss."))
bx, by, bw, bh = 1.2, 2.5, 3.6, 3.3
box(s, bx, by, bw, bh, fill=RGBColor(0xF4, 0xEF, 0xE6), shape=MSO_SHAPE.RECTANGLE, line=INK2)
box(s, bx + 0.05, by + 1.3, bw - 0.1, bh - 1.35, fill=RGBColor(0xBF, 0xD8, 0xF5), shape=MSO_SHAPE.RECTANGLE)
ln = line(s, bx, by + 1.95, bx + bw, by + 1.95, color=RGBColor(0xC0, 0x39, 0x2B), width=2, dash=True)
text(s, bx + 0.15, by + 2.0, 3.3, 0.4, "stress line (RAW used up)", size=12, color=RGBColor(0xC0, 0x39, 0x2B))
text(s, bx, by + bh + 0.1, bw, 0.4, "root zone (total available water, TAW)", size=12, color=INK2,
     align=PP_ALIGN.CENTER)
arrow(s, bx + 0.7, 1.6, bx + 0.7, by + 0.5, color=WATER, width=3)
text(s, bx - 0.1, 1.3, 1.8, 0.3, "rain + irrigation", size=13, color=WATER, bold=True)
arrow(s, bx + 2.8, by + 0.5, bx + 2.8, 1.6, color=GREEN, width=3)
text(s, bx + 2.2, 1.3, 2.4, 0.3, "crop water use (ET)", size=13, color=GREEN, bold=True)
arrow(s, bx + bw, by + bh - 0.4, bx + bw + 1.0, by + bh - 0.4, color=INK2, width=3)
text(s, bx + bw + 0.1, by + bh - 0.3, 1.6, 0.6, "overflow drains away (waste)", size=12, color=INK2)
eqs = [("Crop demand", "ETc = Kc(stage) × ET0"),
       ("Stress factor", "Ks = 1 if depletion ≤ RAW, else falls linearly to 0"),
       ("Actual use", "ETa = Ks × ETc"),
       ("Season yield (FAO-33)", "Ya / Ym = 1 − Ky · (1 − ΣETa / ΣETc)")]
for i, (k, v) in enumerate(eqs):
    y = 1.7 + i * 1.15
    badge(s, 7.1, y, str(i + 1), fill=GREEN)
    text(s, 7.9, y - 0.05, 4.9, 0.4, k, size=16, bold=True, color=GREEN_DK)
    text(s, 7.9, y + 0.35, 4.9, 0.5, v, size=16, color=INK, font="Cambria Math")
text(s, 7.1, 6.3, 5.8, 0.6, "Crop values (Kc, root depth, p, Ky) from FAO-56 / FAO-33 tables for maize, wheat "
     "and tomato; sandy-loam soil.", size=12, color=INK2)

# ================================================================== 9. MDP
s = new_slide("Formulating irrigation as a Markov Decision Process", tag=M1, tag_color=WATER, notes=(
    "Here is the exact formulation. The state is four things a farmer would look at: how dry the soil is, the "
    "crop's growth stage, whether rain is forecast in the next three days, and how hot and dry today is. "
    "Discretised, that is 360 states. The actions are five irrigation amounts. The reward is negative: a cost for "
    "every millimetre of water, a ten times higher cost for every millimetre the crop wanted but didn't get, a "
    "cost for water that drains away, and a small fixed cost per irrigation for the pump and labour. The "
    "ten-to-one ratio encodes the priority that losing yield is far worse than spending water."))
mdp = [
    ("S", "State: what the agent sees", WATER,
     ["Soil dryness: 10 levels", "Crop growth stage: 4", "Rain forecast, next 3 days: none / light / heavy",
      "Evaporative demand (ET0): low / medium / high", ("= 360 states", {"bold": True, "color": WATER})]),
    ("A", "Action: what it decides", GREEN,
     [f"Irrigate {', '.join(map(str, IRRIGATION_ACTIONS_MM))} mm today", "Realistic drip / sprinkler doses",
      ("5 actions", {"bold": True, "color": GREEN})]),
    ("R", "Reward: how it is scored", RUST,
     [f"− {rw.water_cost:g} per mm of water used", f"− {rw.stress_cost:g} per mm of crop water shortage",
      f"− {rw.waste_cost:g} per mm drained away", f"− {rw.event_cost:g} per irrigation event (pump, labour)",
      ("Yield matters 10× more than water", {"bold": True, "color": RUST})]),
]
for i, (b, title, col, items) in enumerate(mdp):
    x = 0.6 + i * 4.1
    box(s, x, 1.6, 3.8, 4.85)
    badge(s, x + 0.3, 1.85, b, fill=col, d=0.65, size=20)
    text(s, x + 1.1, 1.9, 2.6, 0.6, title, size=17, bold=True, color=GREEN_DK)
    text(s, x + 0.3, 2.8, 3.3, 3.5, items, size=15, color=INK, spacing=10)
text(s, 0.6, 6.65, 12, 0.4, "Episode = one crop season (125-145 days) from a random date, city and starting soil "
     "moisture. Forecast = real future rain ± 25 % error.", size=13, color=INK2)

# ================================================================== 10. Q-learning
s = new_slide("Learning: tabular Q-learning", tag=M1, tag_color=WATER, notes=(
    "We used tabular Q-learning, the classic algorithm by Watkins. The agent keeps a table with one value per "
    "state and action: how good is it to give 20 mm when the soil is 70 percent dry and no rain is coming. After "
    "every simulated day it nudges that value towards the reward it got plus the value of the best action in the "
    "next state; that is this update rule. Early on it explores randomly; epsilon decays from 1 to 0.05. We also "
    "decay the learning rate so the values settle. One agent per crop, 50 thousand seasons each, about a minute "
    "and a half on a laptop. The curves rise and then flatten: the agent has converged. We chose a table rather "
    "than a deep network on purpose: 360 states is small, and a table can be printed and inspected."))
box(s, 0.6, 1.55, 6.1, 1.3, fill=TINT)
text(s, 0.6, 1.55, 6.1, 1.3, "Q(s,a) ← Q(s,a) + α [ r + γ · max Q(s′,a′) − Q(s,a) ]", size=18,
     font="Cambria Math", color=GREEN_DK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
hp = [("Q-table", "360 states × 5 actions"), ("Episodes", f"{tc.episodes:,} seasons per crop"),
      ("Exploration ε", f"{tc.eps_start:g} → {tc.eps_end:g}"),
      ("Learning rate α", f"{tc.alpha_start:g} → {tc.alpha_end:g}"),
      ("Discount γ", f"{tc.gamma:g}"), ("Training time", "≈ 1.5 min per crop (CPU)")]
for i, (k, v) in enumerate(hp):
    y = 3.15 + i * 0.5
    text(s, 0.8, y, 2.4, 0.4, k, size=15, bold=True, color=GREEN_DK)
    text(s, 3.2, y, 3.5, 0.4, v, size=15, color=INK)
image(s, RES / "fig_training_curves.png", 7.0, 1.55, w=5.9, h=2.3)
text(s, 7.0, 4.1, 5.9, 2.5, [
    ("Why a table and not a deep network?", {"bold": True, "color": GREEN_DK, "size": 16}),
    "360 states is small enough to learn exactly",
    "Trains in minutes on a laptop, no GPU",
    "The whole policy can be printed and checked, so every decision is explainable",
], size=14, color=INK, spacing=6)

# ================================================================== 11. fair test
s = new_slide("A fair test: four policies on seasons the agent never saw", tag=M1, tag_color=WATER, notes=(
    "To know whether the agent is any good, we compare it with three baselines. Rainfed means never irrigate. "
    "Fixed schedule is the typical farmer calendar. The sensor threshold rule is the strong one: it is the "
    "FAO-recommended practice, it has a perfect soil moisture sensor, and it refills the soil exactly when the "
    "safe limit is reached. It just doesn't use the forecast. All four run on the same 60 test seasons from 2021 "
    "to 2024, with identical weather and identical forecast errors."))
pol = [(RL, "Our agent. Uses soil, crop stage, forecast and ET0."),
       (THR, "FAO expert rule. Perfect moisture sensor; refills to full when the safe limit is reached; no forecast."),
       (FIX, "Farmer's calendar: 40 mm every 7 days, whatever the weather."),
       (RAIN, "Never irrigate. Lower bound.")]
for i, (name, desc) in enumerate(pol):
    x = 0.6 + (i % 2) * 4.3
    y = 1.6 + (i // 2) * 2.35
    box(s, x, y, 4.0, 2.1)
    badge(s, x + 0.3, y + 0.3, "", fill=POLICY[name], d=0.4)
    text(s, x + 0.9, y + 0.28, 3.0, 0.5, name, size=17, bold=True, color=GREEN_DK)
    text(s, x + 0.3, y + 0.9, 3.5, 1.2, desc, size=14, color=INK2)
box(s, 9.5, 1.6, 3.3, 4.45, fill=GREEN_DK)
text(s, 9.8, 1.9, 2.8, 1.0, str(n_test), size=60, bold=True, font=HEAD, color=WHITE)
text(s, 9.8, 2.95, 2.8, 3.0, ["unseen test seasons", "", "3 cities × 5 crop-season combinations × 4 years "
                              "(2021-2024)", "", "Same weather and forecast errors for every policy"],
     size=14, color=WHITE)
text(s, 0.6, 6.4, 12, 0.5, "Metrics: irrigation water · yield (% of max) · water wasted · stress days · "
     "water productivity (kg of crop per m³ of water)", size=14, color=INK2)

# ================================================================== 12. irrigation results
s = new_slide(f"Result: {save_fix:.0f} % less water than the calendar, higher yield", tag=M1, tag_color=WATER,
              notes=(
    f"Here are the headline numbers, averaged over all {n_test} test seasons. Compared with the fixed calendar, the "
    f"agent uses {save_fix:.0f} percent less water, about {s_rl.avg_irrigation_mm:.0f} millimetres instead of "
    f"{s_fix.avg_irrigation_mm:.0f}, and the yield goes up, from {s_fix.avg_relative_yield_pct:.1f} to "
    f"{s_rl.avg_relative_yield_pct:.1f} percent of the maximum. Against the expert sensor rule it is roughly a tie: "
    f"{save_thr:.1f} percent less water for {100 - s_rl.avg_relative_yield_pct:.1f} points of yield. It produces the "
    "most crop per cubic metre of water of any policy. The key point is that the agent was never told the FAO "
    "rule; it reached an expert-level policy purely from trial and error. Its weakest city is Ludhiana, where "
    "yield drops to about 95 percent; we come back to that in the limitations."))
stats_ = [(f"{save_fix:.0f} %", "less water than the fixed calendar", WATER),
          (f"{s_rl.avg_relative_yield_pct:.1f} %", f"of maximum yield (calendar: {s_fix.avg_relative_yield_pct:.1f} %)",
           GREEN_DK),
          (f"{s_rl.water_productivity_kg_per_m3:.2f}", f"kg of crop per m³ of water, the best of all "
                                                       f"(sensor rule {s_thr.water_productivity_kg_per_m3:.2f})",
           GREEN_DK)]
for i, (v, l, c) in enumerate(stats_):
    stat(s, 0.6, 1.5 + i * 1.75, 3.4, v, l, value_color=c, size=48)
image(s, RES / "fig_policy_comparison.png", 4.3, 1.55, w=8.6, h=3.4)
box(s, 4.5, 5.3, 8.3, 1.35)
text(s, 4.8, 5.4, 7.8, 1.15, [("Against the expert sensor rule: ", {"bold": True, "color": GREEN_DK}),
                             f"{save_thr:.1f} % less water and less waste, for a "
                             f"{100 - s_rl.avg_relative_yield_pct:.1f}-point yield trade-off. The agent reached expert "
                             "level without being told the rule."], size=15, color=INK)

# ================================================================== 13. one season
s = new_slide("One season: smaller doses that leave room for rain", tag=M1, tag_color=WATER, notes=(
    "This is one real test season: maize in Jodhpur, monsoon 2023. The top panel is soil moisture; the dashed "
    "line is where stress begins. The orange sensor rule waits until the line is reached, then fills the soil all "
    "the way up with about 55 millimetres. The blue agent gives smaller doses of 30 to 40 millimetres, which "
    "leaves room in the soil for any rain that follows, and it holds back when rain is forecast. That season it "
    f"used {trace_rl:.0f} millimetres against {trace_thr:.0f} for the sensor rule, "
    f"{100 * (1 - trace_rl / trace_thr):.0f} percent less."))
image(s, RES / "fig_season_trace.png", 0.6, 1.4, w=8.0, h=5.6)
notes_ = [("Sensor rule", "waits for the stress line, then refills to 100 %, so later rain can overflow"),
          ("RL agent", "smaller 30-40 mm doses keep room in the soil for rain"),
          ("This season", f"{trace_rl:.0f} mm vs {trace_thr:.0f} mm: {100 * (1 - trace_rl / trace_thr):.0f} % "
                          "less water")]
for i, (h_, d) in enumerate(notes_):
    y = 1.8 + i * 1.6
    badge(s, 9.0, y, str(i + 1), fill=POLICY[THR] if i == 0 else WATER)
    text(s, 9.75, y - 0.02, 3.2, 0.4, h_, size=17, bold=True, color=GREEN_DK)
    text(s, 9.75, y + 0.4, 3.2, 1.0, d, size=14, color=INK2)

# ================================================================== 14. rulebook
s = new_slide("What the agent learned can be read like a rulebook", tag=M1, tag_color=WATER, notes=(
    "Because the policy is a table, we can open it. This is maize on a hot day. Rows are soil dryness, columns "
    "are the rain forecast. Three rules emerge that nobody programmed: don't irrigate until about 60 percent of "
    "the soil water is used, which is almost exactly the FAO threshold of 55 percent; when it's dry and no rain "
    "is coming, give the full 40 millimetres; and when heavy rain is forecast, give less or nothing. Cells marked "
    "rare are situations seen fewer than 50 times in training, so we don't read anything into them."))
image(s, RES / "fig_policy_heatmap.png", 0.5, 1.45, w=12.3, h=4.35)
rules = ["Wait until ~60 % of soil water is used (FAO says 55 %): found without being told",
         "Dry soil and no rain coming: apply the full 40 mm",
         "Heavy rain forecast: irrigate less or not at all"]
for i, r in enumerate(rules):
    x = 0.6 + i * 4.1
    badge(s, x, 6.05, str(i + 1), fill=WATER, d=0.5, size=14)
    text(s, x + 0.65, 6.0, 3.3, 0.9, r, size=14, color=INK)

# ================================================================== 15. module 2 pipeline + why CNN
s = new_slide("Module 2 · Leaf disease detection with a CNN", tag=M2, notes=(
    "Module two diagnoses leaf diseases. The pipeline: a photo of a leaf is resized to 224 by 224 pixels and "
    "normalised. A MobileNetV2 convolutional network, pretrained on ImageNet, turns it into 1280 feature maps "
    "describing textures, spots and colours. Global average pooling squeezes each map to one number, and a "
    f"linear classifier maps those 1280 numbers to our {n_classes} classes. A softmax gives probabilities; we show "
    "the top three, plus a class activation heatmap computed from the same feature maps, and attach advice for "
    "the predicted disease. Why a CNN? Disease is a visual pattern, and CNNs are the standard tool for image "
    "recognition. Why transfer learning? Training from scratch needs millions of images; starting from ImageNet "
    "weights, the network already knows edges and textures, so a few thousand leaf photos are enough."))
nodes = [("Leaf photo", "from phone or upload", TINT),
         ("Pre-process", f"resize {input_size}×{input_size}, normalise", TINT),
         ("MobileNetV2", "pretrained feature extractor", GREEN),
         ("Global avg pool", "1280 features", TINT),
         ("Linear classifier", f"{n_classes} classes", TINT),
         ("Softmax", "top-3 + heatmap", TINT),
         ("Diagnosis", "disease, confidence, advice", GREEN_DK)]
nw, gap = 1.52, 0.27
chain(s, nodes, 0.55, 1.6, nw, 1.45, gap, color=GREEN, size=14, sub_size=11.5)
for i in (2, 6):
    node(s, 0.55 + i * (nw + gap), 1.6, nw, 1.45, nodes[i][0], nodes[i][1], fill=nodes[i][2], title_color=WHITE,
         sub_color=WHITE, size=14, sub_size=11.5)
text(s, 0.55, 3.25, 12.3, 0.4, "Feature maps are reused to draw the heatmap (class activation map): "
     "classifier weights × feature maps = where the model looked.", size=13, color=INK2, italic=True)
text(s, 0.6, 3.95, 12, 0.45, "Why a CNN, and why transfer learning?", size=20, bold=True, color=GREEN_DK, font=HEAD)
whys = [("Visual pattern", "disease shows as spots, rings and colour changes: exactly what convolutions detect"),
        ("Transfer learning", "ImageNet weights already know edges and textures, so thousands of leaf photos suffice"),
        ("Small and fast", "MobileNetV2 has ~3.5 M parameters: small enough to run in a phone's browser"),
        ("Explainable", "the heatmap shows which part of the leaf drove the prediction")]
for i, (h_, d) in enumerate(whys):
    x = 0.6 + i * 3.1
    box(s, x, 4.55, 2.9, 2.15)
    badge(s, x + 0.25, 4.75, str(i + 1), fill=GREEN, d=0.5, size=14)
    text(s, x + 0.9, 4.8, 1.95, 0.5, h_, size=16, bold=True, color=GREEN_DK)
    text(s, x + 0.25, 5.45, 2.5, 1.2, d, size=13.5, color=INK2)

# ================================================================== 16. leaf data
s = new_slide("Data: PlantVillage leaf images, maize and tomato", tag=M2, notes=(
    f"For data we used PlantVillage, the standard public benchmark of labelled leaf photos. We kept only maize and "
    f"tomato, the two crops our irrigation module also covers, giving {n_classes} classes including the healthy ones. "
    f"That is {fmt_int(n_total)} images in total, split into training, validation and test sets; the test images "
    "are never seen during training. Wheat is not in PlantVillage, which is why the leaf module covers maize and "
    "tomato only."))
picture(s, RES / "fig_leaf_dataset.png", 0.55, 1.5, 6.6, 4.1, "Sample leaf images per class")
crops_ = {}
for crop, name in leaf_classes:
    crops_.setdefault(crop, []).append(name)
cx = 7.5
for j, (crop, names) in enumerate(crops_.items()):
    col_x = cx + (0 if j == 0 else 2.45)
    text(s, col_x, 1.5, 2.4, 0.4, f"{crop} ({len(names)})", size=17, bold=True, color=GREEN_DK)
    text(s, col_x, 1.95, 2.35, 3.8, [(n, {"color": GREEN if n.lower().startswith("healthy") else INK,
                                          "bold": n.lower().startswith("healthy")}) for n in names],
         size=12.5, spacing=3)
splits = [("Total images", fmt_int(n_total)), ("Train", fmt_int(_n.get("train"))),
          ("Validation", fmt_int(_n.get("val"))), ("Test (unseen)", fmt_int(_n.get("test")))]
for i, (k, v) in enumerate(splits):
    x = 0.6 + i * 3.1
    box(s, x, 5.85, 2.9, 0.95)
    text(s, x + 0.25, 5.9, 2.5, 0.45, v, size=22, bold=True, color=GREEN_DK, font=HEAD)
    text(s, x + 0.25, 6.38, 2.5, 0.35, k, size=12.5, color=INK2)

# ================================================================== 17. training recipe
ep_head, ep_ft = _ep.get("head"), _ep.get("finetune")
s = new_slide("Training: transfer learning in two phases", tag=M2, notes=(
    "Training happens in two phases. In phase one the pretrained MobileNetV2 body is frozen and only a new "
    f"{n_classes}-class classifier on top is trained, so we don't destroy what the network already knows. In phase "
    "two we unfreeze the later layers and fine-tune them with a small learning rate, so the features adapt to "
    "leaves. We pick the model using the validation set and report accuracy only on the untouched test set. The "
    "finished model is exported to ONNX, a standard format that runs in the browser."))
for i, (b, title, sub, fill, tcol) in enumerate([
        ("1", "Phase 1: train the head", f"backbone frozen, new {n_classes}-class classifier"
         + (f" · {ep_head} epochs" if ep_head else ""), TINT, GREEN_DK),
        ("2", "Phase 2: fine-tune", "unfreeze later layers, small learning rate"
         + (f" · {ep_ft} epochs" if ep_ft else ""), TINT, GREEN_DK),
        ("3", "Pick and test", "best epoch on validation, final score on the untouched test set", TINT, GREEN_DK),
        ("4", "Export", f"ONNX file ({size_txt}) for ONNX Runtime Web", GREEN, WHITE)]):
    x = 0.6 + i * 3.15
    node(s, x, 1.6, 2.8, 1.7, title, sub, fill=fill, title_color=tcol, sub_color=tcol if fill == GREEN else INK2,
         size=15, sub_size=12.5)
    if i:
        arrow(s, x - 0.33, 2.45, x - 0.04, 2.45, color=GREEN)
facts = [("Model", leaf.get("model", "MobileNetV2 (ImageNet-pretrained)")),
         ("Input", f"{input_size} × {input_size} RGB, ImageNet normalisation"),
         ("Classes", f"{n_classes} (maize + tomato, incl. healthy)"),
         ("Epochs", f"{ep_head if ep_head else '—'} head + {ep_ft if ep_ft else '—'} fine-tune"),
         ("Training time", f"{train_min:.0f} min on a laptop CPU" if isinstance(train_min, (int, float)) else "—"),
         ("Model file", size_txt)]
for i, (k, v) in enumerate(facts):
    x = 0.6 + (i % 2) * 6.2
    y = 3.75 + (i // 2) * 0.6
    text(s, x, y, 2.0, 0.4, k, size=15, bold=True, color=GREEN_DK)
    text(s, x + 2.0, y, 4.1, 0.4, str(v), size=15, color=INK)
box(s, 0.6, 5.75, 12.1, 1.0, fill=GREEN_DK)
text(s, 0.95, 5.75, 11.5, 1.0, [("Why two phases?  ", {"bold": True, "color": LEAF}),
                               ("Training the new head first protects the pretrained features; fine-tuning then adapts "
                                "them to leaf textures without forgetting what they already know.",
                                {"color": WHITE})], size=15, anchor=MSO_ANCHOR.MIDDLE)

# ================================================================== 18. leaf results
s = new_slide("Result: " + (f"{acc_txt} accuracy on unseen leaf images" if acc_txt != "—"
                            else "how well it diagnoses unseen leaf images"), tag=M2, notes=(
    f"On the test set, images the model never saw in training, it reaches {acc_txt} accuracy and a macro F1 of "
    f"{f1_txt}. Macro F1 averages the F1 score over all {n_classes} classes equally, so a rare class counts as much "
    "as a common one; that matters because PlantVillage is imbalanced. The confusion matrix shows where mistakes "
    "happen: almost everything sits on the diagonal, and the few errors are between diseases that look similar."
    + (f" The weakest class is {worst.get('crop', '')} {worst.get('name', '')} with an F1 of {worst.get('f1', 0):.2f}."
       if worst else "")))
stat(s, 0.6, 1.5, 3.6, acc_txt, "test accuracy on unseen images", value_color=GREEN, size=50)
stat(s, 0.6, 3.15, 3.6, f1_txt, "macro F1 (all classes weighted equally)", value_color=GREEN_DK, size=50)
stat(s, 0.6, 4.8, 3.6, fmt_int(n_test_img), "test images, never seen in training", value_color=GREEN_DK, size=50)
picture(s, RES / "fig_leaf_confusion.png", 4.5, 1.45, 8.3, 5.4, "Confusion matrix on the test set")

# ================================================================== 19. heatmap + caveat
s = new_slide("Explainable: the heatmap shows where the model looked", tag=M2, notes=(
    "A diagnosis is more trustworthy if we can see why. For every prediction we draw a class activation map: the "
    "classifier's weights for the predicted class multiplied with the network's last feature maps. Bright areas "
    "are the parts of the leaf that pushed the model towards that disease, and they should sit on the lesions, "
    "not on the background. One honest caveat: PlantVillage photos are taken in a lab on plain backgrounds, so "
    "real field photos, with soil, other leaves and harsh light, will be harder, and accuracy there will be lower. "
    "Fine-tuning on field datasets such as PlantDoc is the natural next step."))
picture(s, RES / "fig_leaf_cam_examples.png", 0.55, 1.45, 8.0, 5.3, "Leaf photos with class-activation heatmaps")
pts = [("How it is computed", "classifier weights for the predicted class × last convolutional feature maps"),
       ("How to read it", "bright = the region that drove the prediction; it should sit on the lesions"),
       ("Why it matters", "a farmer or expert can check the model is looking at the disease, not the background")]
for i, (h_, d) in enumerate(pts):
    y = 1.55 + i * 1.25
    badge(s, 8.9, y, str(i + 1), fill=GREEN, d=0.5, size=14)
    text(s, 9.55, y - 0.02, 3.3, 0.4, h_, size=16, bold=True, color=GREEN_DK)
    text(s, 9.55, y + 0.38, 3.3, 0.85, d, size=13, color=INK2)
box(s, 8.9, 5.35, 3.9, 1.4, fill=RGBColor(0xFB, 0xEE, 0xE6))
text(s, 9.1, 5.42, 3.55, 1.28, [("Honest caveat", {"bold": True, "color": RUST, "size": 14}),
                               ("PlantVillage = lab photos on plain backgrounds. Field photos will be harder.",
                                {"color": INK, "size": 13})], anchor=MSO_ANCHOR.MIDDLE, spacing=2)

# ================================================================== 20. website architecture
s = new_slide("The website: no server, everything runs in your browser", tag=WEB, tag_color=INK2, notes=(
    "Both modules are deployed as one static website on GitHub Pages. Static means the server only hands out "
    "files: the page, the JavaScript, the agent's Q-tables and precomputed seasons as JSON, and the leaf model as "
    "an ONNX file. All the computing happens in the visitor's browser. The irrigation pages read the JSON and draw "
    "the seasons and the rulebook. The Leaf Doctor runs the neural network with ONNX Runtime Web, which uses "
    "WebAssembly, so a photo is processed on the phone itself and never uploaded anywhere. Hosting is free, there "
    "is nothing to maintain, and it works on any device with a browser."))
box(s, 0.6, 1.65, 3.4, 4.4, fill=GREEN_DK, radius=0.08)
text(s, 0.85, 1.85, 2.9, 0.5, "GitHub Pages", size=20, bold=True, color=WHITE, font=HEAD)
text(s, 0.85, 2.55, 2.9, 3.4, [("static files only", {"color": LEAF, "bold": True}),
                              ("index.html + JavaScript", {"color": WHITE}),
                              ("Q-tables + seasons (JSON)", {"color": WHITE}),
                              ("weather + results (JSON)", {"color": WHITE}),
                              (f"leaf model (ONNX, {size_txt})", {"color": WHITE})], size=14, spacing=8)
arrow(s, 4.1, 3.85, 5.0, 3.85, color=GREEN, width=3)
text(s, 3.95, 3.35, 1.3, 0.4, "download", size=12, color=INK2, align=PP_ALIGN.CENTER)
box(s, 5.1, 1.65, 7.7, 4.4, fill=TINT, radius=0.05)
text(s, 5.35, 1.78, 7.2, 0.4, "Visitor's browser (phone or laptop)", size=17, bold=True, color=GREEN_DK)
node(s, 5.35, 2.35, 3.5, 1.6, "Irrigation pages", "read JSON, draw seasons, look up the Q-table rulebook",
     fill=BLUE_TINT, title_color=GREEN_DK, size=15, sub_size=12.5)
node(s, 9.05, 2.35, 3.5, 1.6, "Leaf Doctor", "photo → ONNX Runtime Web (WebAssembly) → top-3 + heatmap",
     fill=WHITE, title_color=GREEN_DK, size=15, sub_size=12.5)
for i, (h_, d) in enumerate([("Private", "photos never leave the device"), ("Free", "no server or GPU to pay for"),
                             ("Anywhere", "works on any phone with a browser")]):
    x = 5.35 + i * 2.4
    badge(s, x, 4.35, str(i + 1), fill=GREEN, d=0.45, size=13)
    text(s, x + 0.55, 4.35, 1.75, 0.4, h_, size=15, bold=True, color=GREEN_DK)
    text(s, x + 0.55, 4.75, 1.75, 1.0, d, size=12.5, color=INK2)
text(s, 0.6, 6.35, 12.1, 0.5, "Live at  " + SITE_URL, size=18, bold=True, color=GREEN)

# ================================================================== 21. website tour
s = new_slide("A quick tour of the website", tag=WEB, tag_color=INK2, notes=(
    "A quick tour before the live demo. The home page introduces both modules. The irrigation simulator lets you "
    "pick a crop, city, season and year and compare all four policies day by day. The Leaf Doctor takes a photo and "
    "returns the top three diagnoses with confidence, a heatmap and advice. And the agent page shows the learned "
    "rulebook for any growth stage and weather condition."))
shots = [("home.png", "Home: both modules at a glance"), ("irrigation.png", "Irrigation simulator"),
         ("leaf.png", "Leaf Doctor: diagnosis + heatmap"), ("agent.png", "Inside the agent: the rulebook")]
for i, (fn, cap) in enumerate(shots):
    x = 0.6 + (i % 2) * 6.15
    y = 1.4 + (i // 2) * 2.85
    picture(s, SCREENS / fn, x, y, 5.95, 2.4, cap, frame=True, fill=True)
    text(s, x, y + 2.43, 5.95, 0.35, cap, size=12.5, bold=True, color=GREEN_DK, align=PP_ALIGN.CENTER)

# ================================================================== 22. live demo
s = new_slide(dark=True, notes=(
    "Now the live demo. Open the link or scan the QR code. First, the irrigation simulator: pick Jodhpur, maize, "
    "2023 and show how the agent's soil moisture line stays above the stress line with less water. Second, the "
    "rulebook: change the forecast column and point out the agent waters less when rain is coming. Third, the "
    "Leaf Doctor: load a sample tomato leaf, show the top-3 diagnosis and the heatmap sitting on the lesions."))
text(s, 0.9, 0.8, 8, 0.5, "LIVE DEMO", size=16, bold=True, color=LEAF)
text(s, 0.9, 1.3, 8.5, 1.0, "Try it on your phone", size=44, bold=True, font=HEAD, color=WHITE)
text(s, 0.9, 2.5, 8.6, 0.8, SITE_URL.replace("https://", ""), size=30, bold=True, color=LEAF)
for i, d in enumerate(["Irrigation simulator: Jodhpur, maize, 2023: agent vs sensor rule",
                       "Inside the agent: watch it hold back when rain is forecast",
                       "Leaf Doctor: diagnose a sample tomato leaf, see the heatmap"]):
    badge(s, 0.9, 3.75 + i * 0.85, str(i + 1), fill=WATER if i < 2 else GREEN, d=0.5, size=14)
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
    text(s, 9.75, 4.6, 3.0, 0.4, "scan to open", size=14, color=LEAF, align=PP_ALIGN.CENTER)
except Exception:  # QR is a nice-to-have; never fail the build over it
    pass

# ================================================================== 23. limitations
s = new_slide("Limitations and next steps", notes=(
    "To be honest about limitations. For irrigation: it is a simulated field, not a field trial; the yield model "
    "doesn't know that stress at flowering hurts more; the forecast is synthetic; and the table's coarse states "
    "are why Ludhiana winters are weaker. For leaf disease: PlantVillage photos are lab images on plain "
    "backgrounds, it covers only maize and tomato, and the heatmap is coarse. Next steps: a deep Q-network on "
    "continuous inputs, real soil-moisture sensors, fine-tuning the CNN on field photos, more crops, and linking "
    "the two modules, for example adjusting irrigation when a disease is detected."))
lim = [("Irrigation", "Simulated field (FAO-56), not a field trial; seasonal yield model; synthetic forecast; "
                      "coarse 360-state table (weaker in Ludhiana winters)"),
       ("Leaf disease", "PlantVillage = lab photos on plain backgrounds; maize + tomato only (no wheat); "
                        "heatmap is coarse; no severity estimate")]
nxt = [("Irrigation", "Deep Q-network on continuous inputs; real forecasts and IoT soil-moisture sensors; "
                      "soil type as an input"),
       ("Leaf disease", "Fine-tune on field photos (e.g. PlantDoc); more crops; severity and treatment dosage"),
       ("Together", "Link the modules: a detected disease adjusts the irrigation plan; offline app")]
for i, (title, items, col, mark) in enumerate([("Limitations", lim, RUST, "!"), ("Next steps", nxt, GREEN, "→")]):
    x = 0.6 + i * 6.2
    box(s, x, 1.55, 5.9, 5.25)
    badge(s, x + 0.3, 1.8, mark, fill=col, d=0.6, size=18)
    text(s, x + 1.1, 1.85, 4.5, 0.5, title, size=22, bold=True, color=GREEN_DK, font=HEAD)
    for j, (h_, d) in enumerate(items):
        y = 2.7 + j * 1.35
        text(s, x + 0.4, y, 5.2, 0.35, h_, size=15, bold=True, color=col)
        text(s, x + 0.4, y + 0.35, 5.2, 0.95, d, size=14, color=INK)

# ================================================================== 24. conclusion
s = new_slide(dark=True, notes=(
    f"To conclude. Module one: a simple, explainable Q-learning agent used {save_fix:.0f} percent less water than "
    f"calendar irrigation with a higher yield, and matched an expert sensor rule while using slightly less water. "
    f"Module two: a MobileNetV2 network diagnoses {n_classes} maize and tomato leaf conditions with {acc_txt} "
    "accuracy on unseen images, and shows where it looked. Both run free, in the browser, at the link on screen. "
    "Thank you, we're happy to take questions."))
text(s, 0.9, 0.8, 11, 0.9, "Conclusion", size=40, bold=True, font=HEAD, color=WHITE)
pts = [(f"{save_fix:.0f} %", "less irrigation water than a fixed calendar, with a higher yield", WATER),
       (f"{s_rl.avg_relative_yield_pct:.1f} %", "of maximum yield kept by the RL agent on unseen seasons", WATER),
       (acc_txt, f"leaf-disease accuracy on unseen images ({n_classes} classes)", LEAF),
       ("0 servers", "both models run in the visitor's browser, free to host", LEAF)]
for i, (v, l, col) in enumerate(pts):
    x = 0.9 + i * 3.05
    text(s, x, 2.0, 2.9, 1.0, v, size=44, bold=True, font=HEAD, color=col)
    text(s, x, 3.05, 2.75, 1.0, l, size=15, color=WHITE)
text(s, 0.9, 5.0, 11.5, 0.9, "Thank you. Questions?", size=36, bold=True, font=HEAD, color=WHITE)
text(s, 0.9, 5.95, 11.5, 0.5, SITE_URL, size=18, color=LEAF, bold=True)

prs.save(OUT)
print("saved", OUT, len(prs.slides), "slides")
