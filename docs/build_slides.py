"""Builds docs/AgriSense_Presentation.pptx (11 slides) from the latest results.   python docs/build_slides.py

Minimal slides: a title, one visual and one takeaway line each. The full talk is in the speaker notes
(View > Notes in PowerPoint). Numbers are read from results/ at build time, never typed in.
"""
import json
import os
import sys
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
from agrisense.config import CROPS, DEFAULT_SOIL, IRRIGATION_ACTIONS_MM, SOILS, TrainConfig  # noqa: E402

RES, DOCS = ROOT / "results", ROOT / "docs"
OUT = Path(os.environ.get("DECK_OUT", DOCS / "AgriSense_Presentation.pptx"))
SITE_URL = "https://nirmit10000.github.io/agrisense/"

# palette: charcoal + teal, with a little amber for "new" things only
CHAR = RGBColor(0x1E, 0x23, 0x28)      # dark slides, headings
TEAL = RGBColor(0x0F, 0x76, 0x6E)      # primary accent
TEAL_DK = RGBColor(0x11, 0x5E, 0x59)
TEAL_TINT = RGBColor(0xE0, 0xF2, 0xEF)
TEAL_LT = RGBColor(0x7F, 0xD1, 0xC6)   # accent on dark slides
CARD = RGBColor(0xF3, 0xF5, 0xF6)
LINE_C = RGBColor(0xDD, 0xE3, 0xE6)
AMBER = RGBColor(0xC9, 0x82, 0x2B)
INK = RGBColor(0x1A, 0x1D, 0x21)
MUTED = RGBColor(0x5D, 0x66, 0x70)
SOFT = RGBColor(0xB8, 0xC2, 0xC8)      # secondary text on dark slides
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HEAD, BODY = "Cambria", "Calibri"
# flowchart stage types (same everywhere): data -> processing -> learned model -> output
F_DATA = dict(fill=CARD, title_color=CHAR, sub_color=MUTED, line_color=LINE_C)
F_PROC = dict(fill=TEAL_TINT, title_color=CHAR, sub_color=MUTED)
F_MODEL = dict(fill=TEAL, title_color=WHITE, sub_color=WHITE)
F_OUT = dict(fill=CHAR, title_color=WHITE, sub_color=TEAL_LT)

# ------------------------------------------------------------------ numbers
summary = pd.read_csv(RES / "summary_by_policy.csv", index_col=0)
evals = pd.read_csv(RES / "eval_results.csv")
RL, THR, FIX, RAIN = summary.index.tolist()
s_rl, s_thr, s_fix = summary.loc[RL], summary.loc[THR], summary.loc[FIX]
n_test = len(evals) // 4
save_fix = 100 * (1 - s_rl.avg_irrigation_mm / s_fix.avg_irrigation_mm)
save_thr = 100 * (1 - s_rl.avg_irrigation_mm / s_thr.avg_irrigation_mm)
tc = TrainConfig()
maize = CROPS["Maize"]
taw_m = SOILS[DEFAULT_SOIL].taw(maize.root_depth_m)
raw_m = maize.p * taw_m

leaf = json.loads((RES / "leaf_metrics.json").read_text(encoding="utf-8"))
acc = 100 * leaf["test_accuracy"]
f1 = leaf["macro_f1"]
n_img = leaf["n_images"]
n_total = sum(n_img.values())
n_classes = len(leaf["classes"])
ft_blocks = int(leaf.get("finetune_blocks", 4))
train_min = leaf["train_minutes"]
weak = sorted(leaf["per_class"], key=lambda c: c["f1"])[:2]

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
W, H = 13.333, 7.5


# ------------------------------------------------------------------ helpers
def text(slide, x, y, w, h, content, size=16, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, italic=False, spacing=None):
    """content: str, or a list of paragraphs; a paragraph is str, (str, opts) or a list of runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    for i, ptxt in enumerate(content if isinstance(content, list) else [content]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if spacing:
            p.space_after = Pt(spacing)
        for rt in (ptxt if isinstance(ptxt, list) else [ptxt]):
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


def box(slide, x, y, w, h, fill=CARD, shape=MSO_SHAPE.ROUNDED_RECTANGLE, line=None, radius=0.08, lw=1.0):
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


def badge(slide, x, y, label, fill=TEAL, d=0.5, size=14, color=WHITE):
    c = box(slide, x, y, d, d, fill=fill, shape=MSO_SHAPE.OVAL)
    tf = c.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), True, color, BODY
    return c


def image(slide, path, x, y, w, h):
    """Fit an image inside the (w, h) box, keeping its aspect ratio, centred."""
    iw, ih = Image.open(path).size
    sc = min(w / iw, h / ih)
    pw, ph = iw * sc, ih * sc
    return slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2),
                                    Inches(pw), Inches(ph))


def _arrowhead(connector):
    from lxml import etree
    from pptx.oxml.ns import qn
    ln = connector.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("h", "med")


def line(slide, x1, y1, x2, y2, color=MUTED, width=1.75, head=False, dash=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    if dash:
        c.line.dash_style = 4
    if head:
        _arrowhead(c)
    return c


def arrow(slide, x1, y1, x2, y2, color=MUTED, width=1.75):
    return line(slide, x1, y1, x2, y2, color=color, width=width, head=True)


def node(slide, x, y, w, h, title, sub=None, fill=CARD, title_color=CHAR, sub_color=MUTED, size=14,
         sub_size=11.5, line_color=None):
    box(slide, x, y, w, h, fill=fill, line=line_color, radius=0.12)
    paras = [(title, {"bold": True, "color": title_color, "size": size, "align": PP_ALIGN.CENTER})]
    if sub:
        paras.append((sub, {"color": sub_color, "size": sub_size, "align": PP_ALIGN.CENTER}))
    text(slide, x + 0.08, y + 0.05, w - 0.16, h - 0.1, paras, anchor=MSO_ANCHOR.MIDDLE, spacing=2)


def chain(slide, nodes, x0, y, w, h, gap, **kw):
    for i, (title, sub, style) in enumerate(nodes):
        x = x0 + i * (w + gap)
        node(slide, x, y, w, h, title, sub, **style, **kw)
        if i:
            arrow(slide, x - gap + 0.05, y + h / 2, x - 0.05, y + h / 2)


def legend(slide, x, y, items, size=11.5, step=2.3):
    for i, (lbl, fill) in enumerate(items):
        xx = x + i * step
        box(slide, xx, y + 0.05, 0.3, 0.2, fill=fill, radius=0.2, line=LINE_C if fill == CARD else None)
        text(slide, xx + 0.4, y, step - 0.45, 0.3, lbl, size=size, color=MUTED)


def takeaway(slide, txt, y=6.55):
    """The one line per slide: what to remember."""
    text(slide, 0.6, y, W - 1.2, 0.5, txt, size=15, color=TEAL_DK, italic=True)


def chart_note(slide, x, y, w, txt, size=12):
    """One reading line under a chart."""
    text(slide, x, y, w, 0.5, [[("How to read it  ", {"bold": True, "color": TEAL}), (txt, {"color": MUTED})]],
         size=size)


def new_slide(title=None, dark=False, notes="", tag=None):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = CHAR if dark else WHITE
    if tag:
        text(s, 0.6, 0.28, 10, 0.3, tag, size=11, bold=True, color=TEAL)
    if title:
        text(s, 0.6, 0.52, W - 1.2, 0.8, title, size=30, bold=True, font=HEAD, color=WHITE if dark else CHAR)
    s.notes_slide.notes_text_frame.text = notes
    n = len(prs.slides)
    if n > 1:
        text(s, W - 1.0, H - 0.42, 0.5, 0.3, str(n), size=10, color=SOFT if dark else MUTED, align=PP_ALIGN.RIGHT)
    return s


def stat(slide, x, y, w, value, label, color=CHAR, size=44, label_color=MUTED, label_size=13):
    text(slide, x, y, w, 0.8, value, size=size, bold=True, font=HEAD, color=color)
    text(slide, x, y + 0.82, w, 0.7, label, size=label_size, color=label_color)


M1 = "MODULE 1 · SMART IRRIGATION · REINFORCEMENT LEARNING"
M2 = "MODULE 2 · LEAF DISEASE DETECTION · DEEP LEARNING"

# =================================================================== 1. title
s = new_slide(dark=True, notes=(
    "Good morning. We are Dushyant Kumar and Nirmit Patil, and our project is AgriSense. Our course theme was "
    "agriculture: intelligent irrigation, crop monitoring and resource optimisation. We built two AI modules. The "
    "first uses reinforcement learning to decide when and how much to irrigate. The second uses a convolutional "
    "neural network to recognise leaf diseases from a photo. Both run on one website, which we will show at the end."))
text(s, 0.9, 1.35, 10, 0.4, "EC-422 ARTIFICIAL INTELLIGENCE · COURSE PROJECT", size=13, bold=True, color=TEAL_LT)
text(s, 0.9, 1.95, 11, 1.3, "AgriSense", size=66, bold=True, font=HEAD, color=WHITE)
text(s, 0.9, 3.15, 11, 0.7, "AI for smart irrigation and crop health", size=28, font=HEAD, color=WHITE)
for i, (b, t) in enumerate([("1", "Reinforcement learning decides when and how much to irrigate"),
                            ("2", "A convolutional neural network diagnoses leaf disease from a photo")]):
    badge(s, 0.9, 4.1 + i * 0.55, b, fill=TEAL, d=0.36, size=11)
    text(s, 1.42, 4.13 + i * 0.55, 10, 0.4, t, size=15, color=SOFT)
text(s, 0.9, 5.75, 6, 0.8, ["Dushyant Kumar  ·  23145030", "Nirmit Patil  ·  23145053"], size=15, color=WHITE)
text(s, 7.0, 6.05, 5.5, 0.4, SITE_URL.replace("https://", ""), size=15, bold=True, color=TEAL_LT,
     align=PP_ALIGN.RIGHT)

# =================================================================== 2. problem
s = new_slide("The problem: two decisions every farmer makes", notes=(
    "Every farmer makes two decisions again and again. First: how much should I irrigate today? Second: is this "
    "leaf diseased, and with what? Today the first is usually made by the calendar, and the second by eye. "
    "The numbers show why that matters. About 90 percent of India's freshwater withdrawals go to agriculture, "
    "according to FAO AQUASTAT. In our own simulation, a fixed weekly irrigation schedule lets about "
    f"{s_fix.avg_wasted_mm:.0f} millimetres per season drain below the roots unused. And FAO estimates that up to 40 "
    "percent of crop production is lost to pests and diseases every year. So our problem statement is: use AI to "
    "recommend daily irrigation that saves water without losing yield, and to diagnose leaf diseases from a photo."))
for i, (b, t, d) in enumerate([("1", "Water", "How much should I irrigate today?"),
                               ("2", "Crop health", "Is this leaf diseased, and with what?")]):
    x = 0.6 + i * 6.15
    box(s, x, 1.55, 5.95, 1.15, fill=TEAL_TINT if i == 0 else CARD)
    badge(s, x + 0.3, 1.87, b, fill=TEAL if i == 0 else CHAR)
    text(s, x + 1.05, 1.68, 4.7, 0.9, [(t, {"bold": True, "size": 13, "color": MUTED}),
                                       (d, {"bold": True, "size": 19, "color": CHAR})], anchor=MSO_ANCHOR.MIDDLE)
stats = [("~90 %", "of India's freshwater withdrawals go to agriculture", "FAO AQUASTAT"),
         (f"{s_fix.avg_wasted_mm:.0f} mm", "of water per season drains away under a fixed weekly schedule",
          "our simulation"),
         ("40 %", "of crop production can be lost to pests and diseases each year", "FAO")]
for i, (v, l, src) in enumerate(stats):
    x = 0.6 + i * 4.1
    text(s, x, 3.15, 3.8, 0.9, v, size=48, bold=True, font=HEAD, color=TEAL if i == 1 else CHAR)
    text(s, x, 4.05, 3.7, 0.7, l, size=14, color=INK)
    text(s, x, 4.75, 3.7, 0.3, src, size=11, color=MUTED, italic=True)
box(s, 0.6, 5.5, W - 1.2, 0.95, fill=CHAR, radius=0.12)
text(s, 0.9, 5.5, W - 1.8, 0.95, [[("Problem statement   ", {"bold": True, "color": TEAL_LT}),
                                   ("Use AI to recommend daily irrigation that saves water without losing yield, and to "
                                    "diagnose leaf diseases from a photo.", {"color": WHITE})]],
     size=16, anchor=MSO_ANCHOR.MIDDLE)

# =================================================================== 3. solution / architecture
s = new_slide("Our solution: two AI modules, one website", notes=(
    "This is the whole system on one slide. The top row is the irrigation module: sixteen years of real daily "
    "weather for Pune, Jodhpur and Ludhiana drive a field simulator, a Q-learning agent practises on it for 50,000 "
    "seasons per crop, and what it learns is a table, our rulebook. The bottom row is the leaf module: PlantVillage "
    "leaf photos are used to fine-tune MobileNetV2, a convolutional network, which we export as a model file. "
    "Both are trained offline in Python. The finished results, the rulebook and the leaf model are then served as "
    "files on a website, and everything runs inside the visitor's browser."))
ys = [1.85, 3.95]
rows = [[("Weather data", "16 years · 3 cities", F_DATA), ("Field simulator", "FAO-56 soil-water balance", F_PROC),
         ("Q-learning agent", "50,000 seasons / crop", F_MODEL), ("Rulebook", "Q-table, 360 × 5", F_PROC)],
        [("Leaf photos", f"PlantVillage · {n_classes} classes", F_DATA), ("Transfer learning", "ImageNet → leaves", F_PROC),
         ("MobileNetV2", "CNN, fine-tuned", F_MODEL), ("Leaf model", "ONNX file", F_PROC)]]
for y, (label, r) in zip(ys, [("Module 1 · irrigation", rows[0]), ("Module 2 · leaf disease", rows[1])]):
    text(s, 0.6, y - 0.42, 4, 0.35, label, size=13, bold=True, color=TEAL)
    chain(s, r, 0.6, y, 2.05, 1.15, 0.45, size=14, sub_size=11.5)
web_x, web_y, web_w, web_h = 10.55, 1.85, 2.2, 3.25
node(s, web_x, web_y, web_w, web_h, "Website", "GitHub Pages\n\nirrigation simulator\n+ rulebook\n\nleaf doctor\n"
     "+ heatmap", size=17, sub_size=12, **F_OUT)
x_end = 0.6 + 4 * 2.05 + 3 * 0.45
for y in ys:
    arrow(s, x_end + 0.05, y + 0.575, web_x - 0.05, y + 0.575, color=TEAL, width=2)
legend(s, 0.6, 5.55, [("data", CARD), ("processing", TEAL_TINT), ("learned model", TEAL), ("output", CHAR)])
takeaway(s, "Both models are trained offline in Python; the website runs them in the browser.")

# =================================================================== 4. RL formulation
s = new_slide("Irrigation as a reinforcement-learning problem", tag=M1, notes=(
    "We turned irrigation into a reinforcement-learning problem. The environment is a field simulator built on "
    "FAO-56, the international standard for crop water use. Think of the root zone as a bucket. Rain and "
    "irrigation fill it, anything above the brim drains away and is wasted, and the crop drinks from it every day "
    "depending on the weather and its growth stage. Below the red line the crop is water-stressed and yield falls. "
    "On the right are the three parts of the problem. The state is what the agent sees: how dry the soil is, the "
    "crop stage, the rain forecast for three days and how hot and dry the day is; that gives 360 situations. The "
    "action is one of five doses, 0 to 40 millimetres. The reward is a cost: every millimetre of water costs 1, "
    "every millimetre of water the crop needed but did not get costs 10, so yield matters ten times more than water. "
    "Water that drains away and each irrigation event are also penalised."))
bx, by, bw, bh = 1.0, 2.45, 3.6, 3.0
box(s, bx, by, bw, bh, fill=CARD, shape=MSO_SHAPE.RECTANGLE, line=MUTED)
box(s, bx + 0.05, by + 1.1, bw - 0.1, bh - 1.15, fill=RGBColor(0xC6, 0xE3, 0xDF), shape=MSO_SHAPE.RECTANGLE)
line(s, bx, by + 1.85, bx + bw, by + 1.85, color=AMBER, width=2, dash=True)
text(s, bx + 0.15, by + 1.9, 3.3, 0.35, "stress line", size=12, color=AMBER, bold=True)
text(s, bx, by + bh + 0.1, bw, 0.35, "root zone of the crop", size=12, color=MUTED, align=PP_ALIGN.CENTER)
arrow(s, bx + 0.7, 1.55, bx + 0.7, by + 0.45, color=TEAL, width=3)
text(s, bx - 0.2, 1.25, 2.0, 0.3, "rain + irrigation", size=13, color=TEAL, bold=True)
arrow(s, bx + 2.8, by + 0.45, bx + 2.8, 1.55, color=CHAR, width=3)
text(s, bx + 2.2, 1.25, 2.4, 0.3, "crop water use", size=13, color=CHAR, bold=True)
arrow(s, bx + bw, by + bh - 0.35, bx + bw + 0.9, by + bh - 0.35, color=MUTED, width=3)
text(s, bx + bw + 0.05, by + bh - 0.25, 1.4, 0.5, "overflow is lost", size=12, color=MUTED)
cards = [("S", "State", "soil dryness · crop stage ·\n3-day rain forecast · ET0", "360 situations", TEAL),
         ("A", "Action", "irrigate 0, 10, 20, 30\nor 40 mm today", "5 choices", TEAL_DK),
         ("R", "Reward", "−(water + 10 × crop shortage\n+ drained water + event cost)", "yield ≫ water", AMBER)]
for i, (b, t, d, k, col) in enumerate(cards):
    y = 1.45 + i * 1.62
    box(s, 6.7, y, 6.05, 1.42)
    badge(s, 6.95, y + 0.42, b, fill=col, d=0.58, size=17)
    text(s, 7.8, y + 0.14, 3.2, 0.4, t, size=17, bold=True, color=CHAR)
    text(s, 7.8, y + 0.52, 3.6, 0.8, d, size=13, color=MUTED)
    text(s, 10.9, y + 0.45, 1.7, 0.5, k, size=15, bold=True, color=col, align=PP_ALIGN.RIGHT)
takeaway(s, "Environment: an FAO-56 field simulator driven by 16 years of real weather from Pune, Jodhpur and Ludhiana.")

# =================================================================== 5. learning and deciding
s = new_slide("How the agent learns and decides", tag=M1, notes=(
    "The algorithm is tabular Q-learning. The agent keeps one number for every situation and every dose: how good is "
    "it to give, say, 40 millimetres when the soil is 70 percent dry and no rain is coming. The rule at the top is how "
    "that number is updated after every simulated day: move it a little, by the learning rate alpha, towards the "
    "reward just received plus the discounted value of the best action tomorrow. The bottom row is one decision. The "
    "agent observes the field, looks up that row of the table, here five illustrative values, and picks the highest, "
    "40 millimetres. During training the simulator then returns the next state and the reward, and the update rule "
    "nudges the value. We trained 50,000 seasons per crop. The curve on the right is the season reward during "
    "training: it rises while the agent explores and then flattens, which shows it converged."))
box(s, 0.6, 1.5, 6.6, 1.25, fill=TEAL_TINT, radius=0.1)
text(s, 0.6, 1.52, 6.6, 0.75, "Q(s,a) ← Q(s,a) + α [ r + γ · max Q(s′,a′) − Q(s,a) ]", size=21,
     font="Cambria Math", color=CHAR, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
text(s, 0.6, 2.25, 6.6, 0.4, f"α {tc.alpha_start:g} → {tc.alpha_end:g}  ·  γ {tc.gamma:g}  ·  ε {tc.eps_start:g} → "
     f"{tc.eps_end:g}  ·  {tc.episodes:,} seasons per crop", size=12.5, color=MUTED, align=PP_ALIGN.CENTER)
image(s, RES / "fig_training_curves.png", 7.5, 1.4, 5.25, 1.75)
chart_note(s, 7.5, 3.2, 5.3, "reward per season rises, then flattens: converged.", size=11.5)
# one decision
yd = 4.05
node(s, 0.6, yd, 2.45, 1.9, "1 · Observe", "soil 70–80 % dry\nmid-season\nno rain forecast\nhot day", size=15,
     sub_size=12, **F_DATA)
arrow(s, 3.1, yd + 0.95, 3.5, yd + 0.95)
box(s, 3.55, yd, 5.15, 1.9, fill=TEAL_TINT, radius=0.08)
text(s, 3.75, yd + 0.1, 4.8, 0.35, "2 · Look up that row of the Q-table", size=15, bold=True, color=CHAR)
for i, (a, v) in enumerate(zip(IRRIGATION_ACTIONS_MM, [-612, -571, -540, -522, -509])):
    x = 3.75 + i * 0.97
    best = i == 4
    box(s, x, yd + 0.55, 0.88, 0.95, fill=TEAL if best else WHITE, radius=0.12, line=None if best else LINE_C)
    text(s, x, yd + 0.6, 0.88, 0.35, f"{a} mm", size=12, bold=True, color=WHITE if best else MUTED,
         align=PP_ALIGN.CENTER)
    text(s, x, yd + 0.95, 0.88, 0.45, str(v), size=16, bold=True, font=HEAD, color=WHITE if best else CHAR,
         align=PP_ALIGN.CENTER)
text(s, 3.75, yd + 1.55, 4.8, 0.3, "illustrative values · pick the highest", size=11.5, color=MUTED, italic=True)
arrow(s, 8.75, yd + 0.95, 9.15, yd + 0.95)
node(s, 9.2, yd, 1.65, 1.9, "3 · Act", "irrigate\n40 mm", size=15, sub_size=13, **F_OUT)
arrow(s, 10.9, yd + 0.95, 11.3, yd + 0.95)
node(s, 11.35, yd, 1.4, 1.9, "4 · Learn", "reward →\nupdate Q\n(training)", size=15, sub_size=12, **F_MODEL)
takeaway(s, "After training, steps 1–3 are just a table look-up: fast, and every decision can be read and explained.")

# =================================================================== 6. benchmark + results
s = new_slide(f"Results: {save_fix:.0f} % less water than a fixed schedule", tag=M1, notes=(
    f"We tested on {n_test} seasons from 2021 to 2024 that the agent never saw during training, in all three cities, "
    "against three policies. Rainfed never irrigates. The fixed schedule gives 40 millimetres every seven days, like "
    "a farmer's calendar. The sensor rule is our strong benchmark: it is the FAO-56 recommended practice, irrigating "
    "as soon as the soil dries past the safe limit and refilling it completely. We even give it a perfect moisture "
    "sensor. Its blind spot is that it ignores the weather forecast. In the chart, the left bars are water used per "
    f"season and the right bars yield. Our agent uses {save_fix:.0f} percent less water than the fixed schedule and "
    f"gets a higher yield, {s_rl.avg_relative_yield_pct:.1f} against {s_fix.avg_relative_yield_pct:.1f} percent. "
    f"Against the expert sensor rule it uses {save_thr:.1f} percent less water for {100 - s_rl.avg_relative_yield_pct:.1f} "
    "points of yield, and it gets the most crop per cubic metre of water. It reached the expert rule's level without "
    "ever being told the rule."))
stat(s, 0.6, 1.45, 3.6, f"{save_fix:.0f} %", "less water than the fixed schedule", color=TEAL, size=46)
stat(s, 0.6, 3.0, 3.6, f"{save_thr:.1f} %", "less water than the expert sensor rule", size=40)
stat(s, 0.6, 4.45, 3.6, f"{s_rl.avg_relative_yield_pct:.1f} %", "of maximum yield kept", size=40)
image(s, RES / "fig_policy_comparison.png", 4.4, 1.45, 8.35, 3.3)
chart_note(s, 4.55, 4.8, 8.2, "left: water used per season (lower is better); right: yield (higher is better). "
           f"Averages over {n_test} unseen seasons.", size=12)
box(s, 4.55, 5.35, 8.2, 1.0, fill=CARD, radius=0.1)
text(s, 4.8, 5.35, 7.8, 1.0, [[("Benchmark · sensor rule (FAO-56)   ", {"bold": True, "color": AMBER}),
                               ("irrigate when the safe soil water is used up, refill to full. Perfect sensor, "
                                "but no forecast.", {"color": INK})]], size=13.5, anchor=MSO_ANCHOR.MIDDLE)
takeaway(s, "The agent matched the expert rule by trial and error, and used the rain forecast to save more water.")

# =================================================================== 7. rulebook
s = new_slide("What the agent learned: a readable rulebook", tag=M1, notes=(
    "Because the policy is a table, we can open it and read it. This is maize on a hot day, for three crop stages. "
    "Each row is a level of soil dryness and each column is the rain forecast; the number is the irrigation the "
    "agent chooses. Three rules appear that nobody programmed. It does not irrigate until about 60 percent of the "
    "soil water is used; the FAO-56 guideline for maize is 55 percent. When the soil is dry and no rain is coming, it "
    "gives the full 40 millimetres. And when heavy rain is forecast, it irrigates less or not at all, which is where "
    "it saves water compared with the sensor rule. Cells marked rare are situations seen fewer than 50 times in "
    "training, so we do not read anything into them."))
image(s, RES / "fig_policy_heatmap.png", 0.6, 1.4, 12.15, 4.2)
chart_note(s, 0.6, 5.65, 12.1, "rows = how dry the soil is, columns = rain forecast; each cell is the dose the agent "
           "picks (darker = more water).", size=12.5)
rules = ["waits until ~60 % of soil water is used (FAO: 55 %)", "dry + no rain coming → full 40 mm",
         "heavy rain forecast → less or nothing"]
for i, r in enumerate(rules):
    x = 0.6 + i * 4.1
    badge(s, x, 6.15, str(i + 1), fill=TEAL, d=0.42, size=12)
    text(s, x + 0.55, 6.17, 3.45, 0.6, r, size=14, color=CHAR, bold=True)

# =================================================================== 8. leaf data + pipeline
s = new_slide("Leaf disease detection: data and pipeline", tag=M2, notes=(
    "The second module recognises leaf diseases. A convolutional neural network is the natural choice, because "
    "disease shows up as visual patterns: spots, rings and discoloured patches. Our data is PlantVillage, a public "
    f"dataset of labelled leaf photos. We used the maize and tomato classes so the module matches our irrigation "
    f"crops: {n_classes} classes including healthy leaves, {n_total:,} images, split 70, 15 and 15 percent into "
    "training, validation and test. The pipeline: a photo is resized and cropped to 224 by 224 pixels and normalised, "
    "MobileNetV2 turns it into 1280 feature maps of 7 by 7, global average pooling and a linear layer give a score for "
    "each class, and softmax turns those into probabilities. We show the top three, plus a heatmap of where the "
    "network looked. Because we had only a few thousand images, we used transfer learning: start from a network "
    "already trained on ImageNet and retrain only its last layers on leaves."))
facts = [(f"{n_classes}", "classes (4 maize, 10 tomato)"), (f"{n_total:,}", "leaf photos, PlantVillage"),
         (f"{n_img['train']:,} / {n_img['val']:,} / {n_img['test']:,}", "train / validation / test")]
for i, (v, l) in enumerate(facts):
    x = 0.6 + i * 4.1
    box(s, x, 1.5, 3.85, 1.35)
    text(s, x + 0.3, 1.58, 3.4, 0.7, v, size=30 if i < 2 else 24, bold=True, font=HEAD, color=TEAL if i == 0 else CHAR,
         anchor=MSO_ANCHOR.MIDDLE)
    text(s, x + 0.3, 2.28, 3.4, 0.45, l, size=13, color=MUTED)
pipe = [("Leaf photo", "camera or upload", F_DATA), ("Pre-process", "224 × 224, normalise", F_PROC),
        ("MobileNetV2", "7 × 7 × 1280 features", F_MODEL), ("Pool + linear", f"1280 → {n_classes} scores", F_PROC),
        ("Diagnosis", "top-3 + heatmap", F_OUT)]
chain(s, pipe, 0.6, 3.5, 2.15, 1.45, 0.35, size=15, sub_size=12)
legend(s, 0.6, 5.3, [("data", CARD), ("processing", TEAL_TINT), ("learned model", TEAL), ("output", CHAR)])
takeaway(s, "Transfer learning: start from a network trained on ImageNet, retrain its last layers on leaf photos.")

# =================================================================== 9. MobileNetV2
s = new_slide("MobileNetV2: the architecture, and why we chose it", tag=M2, notes=(
    "This is the network inside the leaf module, MobileNetV2. A 224 by 224 colour photo passes through one ordinary "
    "convolution and then 17 inverted-residual blocks grouped into seven stages. Going right, the image gets smaller, "
    "from 112 by 112 down to 7 by 7, while the number of channels grows from 16 to 320, and a last 1 by 1 convolution "
    "produces 1280 feature maps. Early layers detect edges and colours, later ones textures and lesion shapes. Global "
    f"average pooling turns each map into one number and our new linear layer maps them to {n_classes} classes. The "
    f"colours show how we trained it: the light blocks keep their ImageNet weights frozen, the last {ft_blocks} blocks "
    "were fine-tuned on leaves, and the amber classifier is new. Why this network? It has about 3.5 million "
    "parameters, against 25 million for ResNet-50, because each block uses depthwise-separable convolutions, which "
    f"are about eight to nine times cheaper. That let us train in about {train_min:.0f} minutes on a laptop CPU, run "
    "it inside a web browser, and, because it ends in pooling plus one linear layer, draw an exact heatmap."))
FEAT = [("Conv 3×3", "32 ch", "112", [0]), ("Stage 1", "16 ch", "112", [1]), ("Stage 2", "24 ch", "56", [2, 3]),
        ("Stage 3", "32 ch", "28", [4, 5, 6]), ("Stage 4", "64 ch", "14", [7, 8, 9, 10]),
        ("Stage 5", "96 ch", "14", [11, 12, 13]), ("Stage 6", "160 ch", "7", [14, 15, 16]),
        ("Stage 7", "320 ch", "7", [17]), ("Conv 1×1", "1280 ch", "7", [18])]
ft_set = set(range(19 - ft_blocks, 19))
HEIGHTS = {"112": 2.2, "56": 1.9, "28": 1.6, "14": 1.35, "7": 1.1}
mid, bw_, gap_ = 2.85, 1.07, 0.16
x = 0.6
node(s, x, mid - 1.1, 0.95, 2.2, "Input", "224×224\nphoto", size=13, sub_size=11, **F_DATA)
x += 0.95 + gap_
for name, ch, res, idx in FEAT:
    n_ft = len(set(idx) & ft_set)
    style = F_MODEL if n_ft == len(idx) else (dict(fill=RGBColor(0x8C, 0xC7, 0xBF), title_color=CHAR, sub_color=CHAR)
                                              if n_ft else F_PROC)
    hh = HEIGHTS[res]
    node(s, x, mid - hh / 2, bw_, hh, name, f"{ch}\n{res}×{res}", size=12, sub_size=10.5, **style)
    arrow(s, x - gap_ + 0.02, mid, x - 0.02, mid, width=1.25)
    x += bw_ + gap_
bx0 = 0.6 + 0.95 + gap_ + (bw_ + gap_)
bx1 = 0.6 + 0.95 + gap_ + 8 * (bw_ + gap_) - gap_
line(s, bx0, 1.52, bx1, 1.52, color=TEAL, width=1.25)
line(s, bx0, 1.52, bx0, 1.63, color=TEAL, width=1.25)
line(s, bx1, 1.52, bx1, 1.63, color=TEAL, width=1.25)
text(s, bx0, 1.22, bx1 - bx0, 0.3, "17 inverted-residual blocks (depthwise-separable convolutions)", size=12,
     color=TEAL, bold=True, align=PP_ALIGN.CENTER)
yh, hw, hg = 4.3, 1.75, 0.3
x_last = x - gap_
hx = x_last - 3 * hw - 2 * hg
xc = x_last - bw_ / 2
line(s, xc, mid + 0.55, xc, yh - 0.25)
line(s, xc, yh - 0.25, hx + hw / 2, yh - 0.25)
arrow(s, hx + hw / 2, yh - 0.25, hx + hw / 2, yh - 0.02)
for i, (t, sub, st) in enumerate([("Global avg pool", "1280 numbers", F_PROC),
                                  ("Linear layer", f"1280 → {n_classes} (new)",
                                   dict(fill=AMBER, title_color=WHITE, sub_color=WHITE)),
                                  ("Softmax", f"{n_classes} probabilities", F_OUT)]):
    xx = hx + i * (hw + hg)
    node(s, xx, yh, hw, 0.85, t, sub, size=13, sub_size=11.5, **st)
    if i:
        arrow(s, xx - hg + 0.03, yh + 0.42, xx - 0.03, yh + 0.42)
legend(s, 0.6, 4.4, [("pretrained, frozen", TEAL_TINT), (f"fine-tuned (last {ft_blocks})", TEAL)], step=2.45)
legend(s, 0.6, 4.8, [("new, trained by us", AMBER), ("input / output", CHAR)], step=2.45)
why = [("~3.5 M", "parameters (ResNet-50: ~25.6 M)"), (f"~{train_min:.0f} min", "to train on a laptop CPU, no GPU"),
       ("In browser", "small and fast enough to run on a phone"), ("Exact heatmap", "pooling + one linear layer")]
for i, (v, l) in enumerate(why):
    xx = 0.6 + i * 3.08
    box(s, xx, 5.45, 2.9, 1.0, fill=CARD)
    text(s, xx + 0.2, 5.5, 2.6, 0.5, v, size=19, bold=True, font=HEAD, color=TEAL, anchor=MSO_ANCHOR.MIDDLE)
    text(s, xx + 0.2, 5.98, 2.6, 0.45, l, size=11.5, color=MUTED)
takeaway(s, "Small, fast and pretrained: the right size for a laptop to train and a phone to run.", y=6.6)

# =================================================================== 10. leaf results
s = new_slide(f"Results: {acc:.1f} % accuracy on unseen leaf images", tag=M2, notes=(
    f"We evaluated on {n_img['test']:,} test images the network never saw. It names the correct class for "
    f"{acc:.1f} percent of them, and the macro F1 score, which weighs every class equally, is {f1:.3f}. The middle "
    "figure is the confusion matrix: each row is the true disease and each column the prediction, so everything on "
    "the diagonal is correct. Almost all of it is on the diagonal. The mistakes cluster in a few look-alike classes: "
    f"{weak[0]['crop'].lower()} {weak[0]['name'].lower()} and {weak[1]['crop'].lower()} {weak[1]['name'].lower()} are "
    "the weakest, because their spots look similar. The right figure shows the heatmaps: the top row is the test "
    "photo, the bottom row the same photo with the class activation map, which is the classifier weights applied to "
    "the last feature maps. The bright area sits on the lesions, not the background, so the network is using the right "
    "evidence. One honest caveat: PlantVillage photos are single leaves on plain backgrounds, so photos from a real "
    "field would be harder."))
image(s, RES / "fig_leaf_confusion.png", 0.5, 1.3, 5.95, 4.95)
chart_note(s, 0.6, 6.25, 5.9, "rows = true class, columns = prediction; the diagonal is correct.", size=12)
for i, (v, l, col) in enumerate([(f"{acc:.1f} %", "test accuracy", TEAL), (f"{f1:.3f}", "macro F1", CHAR),
                                 (f"{n_img['test']:,}", "unseen test images", CHAR)]):
    stat(s, 6.75 + i * 2.05, 1.35, 2.0, v, l, color=col, size=32 if i == 0 else 28, label_size=12.5)
image(s, RES / "fig_leaf_cam_examples.png", 6.7, 2.75, 6.05, 3.3)
chart_note(s, 6.75, 6.1, 6.0, "top: test photo; bottom: heatmap, bright = evidence for the prediction.", size=12)
takeaway(s, "Mistakes are limited to look-alike spot diseases, and the heatmaps sit on the lesions.", y=6.72)

# =================================================================== 11. conclusion
s = new_slide(dark=True, notes=(
    f"To conclude. Our reinforcement-learning agent uses {save_fix:.0f} percent less irrigation water than a fixed "
    f"schedule while keeping {s_rl.avg_relative_yield_pct:.1f} percent of the maximum yield, and our MobileNetV2 "
    f"model identifies {n_classes} maize and tomato leaf classes with {acc:.1f} percent accuracy. Behind those numbers: "
    f"two trained models, {3 * tc.episodes:,} simulated training seasons, sixteen years of weather from three cities, "
    f"{n_total:,} leaf images, {len(evals)} test runs, a live website and a full report. For the future: connect real "
    "soil-moisture sensors and live forecasts to give a daily recommendation, fine-tune the leaf model on field photos "
    "with a GPU, and link the modules so that a detected disease adjusts the irrigation plan. The website is live at "
    "the link shown. Thank you, we are happy to take questions."))
text(s, 0.9, 0.7, 11, 0.8, "Conclusion", size=36, bold=True, font=HEAD, color=WHITE)
heads = [(f"{save_fix:.0f} %", "less irrigation water than a fixed schedule"),
         (f"{s_rl.avg_relative_yield_pct:.1f} %", "of maximum yield kept by the RL agent"),
         (f"{acc:.1f} %", f"leaf-disease accuracy on unseen images ({n_classes} classes)")]
for i, (v, l) in enumerate(heads):
    stat(s, 0.9 + i * 4.0, 1.55, 3.6, v, l, color=TEAL_LT, size=46, label_color=WHITE, label_size=14)
text(s, 0.9, 3.35, 4, 0.35, "WHAT WE BUILT", size=12, bold=True, color=TEAL_LT)
built = [("2", "trained models"), (f"{3 * tc.episodes // 1000}k", "simulated seasons"), ("16 yrs", "weather, 3 cities"),
         (f"{n_total:,}", "leaf images"), (str(len(evals)), "test runs"), ("1", "live website")]
for i, (v, l) in enumerate(built):
    xx = 0.9 + i * 1.95
    text(s, xx, 3.7, 1.85, 0.5, v, size=22, bold=True, font=HEAD, color=WHITE)
    text(s, xx, 4.2, 1.85, 0.35, l, size=12, color=SOFT)
text(s, 0.9, 4.85, 4, 0.35, "FUTURE SCOPE", size=12, bold=True, color=TEAL_LT)
future = ["live soil sensors + forecasts → a daily recommendation", "fine-tune the leaf model on field photos (GPU)",
          "link modules: a detected disease adjusts irrigation"]
for i, f_ in enumerate(future):
    text(s, 0.9 + i * 4.0, 5.2, 3.75, 0.7, f_, size=14, color=WHITE)
line(s, 0.9, 6.05, W - 0.9, 6.05, color=RGBColor(0x3A, 0x42, 0x4A), width=1)
text(s, 0.9, 6.25, 6, 0.6, "Thank you. Questions?", size=26, bold=True, font=HEAD, color=WHITE)
text(s, 6.6, 6.38, 5.85, 0.4, SITE_URL.replace("https://", ""), size=16, bold=True, color=TEAL_LT,
     align=PP_ALIGN.RIGHT)

try:
    prs.save(OUT)
except PermissionError:
    OUT = OUT.with_name(OUT.stem + "_v3.pptx")
    prs.save(OUT)
    print("NOTE: the original file is open in another program; saved a copy instead.")
print("saved", OUT, len(prs.slides), "slides")
