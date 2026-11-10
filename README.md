# AgriSense: AI for Smart Irrigation and Crop Health Monitoring

**EC-422 Artificial Intelligence: course project**
Domain: Agriculture (intelligent irrigation, crop monitoring and resource optimisation)

**Live site: <https://nirmit10000.github.io/agrisense/>**

AgriSense answers a farmer's two everyday questions with two AI techniques:

| Module | Question | AI technique |
|---|---|---|
| **A. Smart irrigation** | When, and how much, should I water? | Reinforcement learning (tabular Q-learning) |
| **B. Leaf Doctor** | Is this leaf healthy, and if not, which disease is it? | Deep learning (MobileNetV2 CNN, transfer learning) + class activation maps |

### A. Smart irrigation

A Q-learning agent learns **when and how much to irrigate** a field by practising in a simulated
field that uses real weather from three Indian climates (Pune, Jodhpur, Ludhiana, 2010-2025).
On 60 test seasons (2021-2024) it never saw during training, it uses **45% less water than a fixed irrigation
schedule** and **5.5% less than a soil-moisture sensor rule**, while keeping **98% of maximum
yield**. It also produces the most crop per cubic metre of water of all the policies.

| Policy (avg over 60 unseen test seasons) | Water / season | Yield (% of max) | Crop per m³ of water |
|---|---|---|---|
| **Q-learning agent (RL)** | **435 mm** | 98.1 % | **6.30 kg** |
| Sensor threshold rule | 460 mm | 100.0 % | 6.13 kg |
| Fixed schedule (40 mm / 7 days) | 784 mm | 93.7 % | 3.33 kg |
| Rainfed (no irrigation) | 0 mm | 31.8 % | - |

### B. Leaf disease detection

A MobileNetV2 convolutional neural network, pre-trained on ImageNet and fine-tuned on the **maize and
tomato** images of the public PlantVillage dataset (14 classes: healthy plus common diseases), classifies a
leaf photo and shows a **class activation map**: a heatmap of the leaf regions that drove the decision.
Test-set accuracy, per-class scores and the confusion matrix are in `results/leaf_metrics.json` and
`results/fig_leaf_confusion.png`. The model is exported to ONNX and runs **in the browser** with ONNX
Runtime Web, so photos never leave the user's device.

> PlantVillage photos are single leaves on plain backgrounds; real field photos are harder, and accuracy
> on them will be lower.

---

## Run it

### The website (both modules)

The site is static (HTML/CSS/JS, no build step). From the repository root:

```bash
python -m http.server
```

Then open <http://localhost:8000>. (Opening `index.html` directly from disk will not work, because the
browser blocks loading the model and JSON files from `file://`.)

### The local Streamlit dashboard (irrigation only)

```bash
pip install -r requirements.txt
streamlit run app.py
```

The trained agents (`models/`), weather data (`data/`) and results (`results/`) are included, so
it runs immediately and needs no internet.

### Re-creating the irrigation results (about 5 minutes on a laptop CPU)

```bash
python -m agrisense.weather    # download weather again (optional, ~0.5 MB)
python -m agrisense.train      # train one agent per crop (Maize, Wheat, Tomato)
python -m agrisense.evaluate   # test on 2021-2024 seasons and redraw all figures
```

The leaf-disease training code is in `leaf/`. The data download and training steps are documented
there.

## Deployment

The site is served by **GitHub Pages** from this repository: `index.html` at the root, assets in `web/`.
It has three tabs:

| Tab | What it shows |
|---|---|
| **Overview** | The problem, the two modules and flowcharts of how data, training and the website connect |
| **Smart irrigation** | Season simulator (crop, city, sowing season, year, policies to plot), the learned rulebook, and results on the 60 unseen test seasons |
| **Leaf doctor** | Upload or photograph a leaf (or pick a test sample): top-3 diagnoses, heatmap and advice, plus the test-set results |

Irrigation simulations for every city, crop, season, year and policy are precomputed in Python and
shipped as JSON, together with the trained Q-tables. The leaf model is an ONNX file executed in the
browser with ONNX Runtime Web. There is no server, so hosting is free and the link always works.

## How the irrigation agent works (one paragraph)

Each day the agent looks at four things: **how dry the soil is** (10 levels), **the crop's growth
stage** (4), **the rain forecast for the next 3 days** (3), and **how hot/dry the day is** (ET0, 3).
That is 360 possible situations. It picks an irrigation amount (0/10/20/30/40 mm). A simulator
built on the FAO-56 soil-water balance then works out what happens to the soil and crop, and
returns a **reward**: minus (water used + 10 × crop water shortage + water wasted to drainage + a
fixed cost per irrigation). Over 50,000 simulated seasons, Q-learning fills in a 360 × 5 table of
"how good is each action in each situation". The final policy is just "pick the best action in
the table", which you can print and read like a rulebook.

## How the leaf model works (one paragraph)

The photo is resized and normalised, then passed through MobileNetV2. Its convolutional layers turn the
image into a grid of feature maps (edges, colours, textures, lesion shapes); global average pooling and
one linear layer turn those into a score per class, and softmax turns scores into probabilities. Training
first fits only the new linear layer on frozen ImageNet features, then fine-tunes the last convolutional
blocks. The heatmap is the classifier-weighted sum of the last feature maps (CAM), which for this
architecture is exactly what the class score is built from.

## Project layout

```
index.html      the website (entry point for GitHub Pages)
web/            website assets: scripts, styles, exported JSON data, the ONNX leaf model
agrisense/      irrigation module (Python)
  config.py     all settings: cities, crops (FAO-56 values), soil, reward weights, training
  weather.py    download + cache daily weather from Open-Meteo (free, no API key)
  env.py        the field simulator (FAO-56 water balance, FAO-33 yield)
  agents.py     Q-learning agent + 3 baseline policies (same act(obs) interface)
  train.py      training loop (one agent per crop)
  simulate.py   run any policy through a season; batch evaluation
  evaluate.py   test-set evaluation, summary tables, report figures
leaf/           leaf-disease module: dataset preparation, MobileNetV2 training, ONNX export
app.py          Streamlit dashboard (local use)
data/           cached weather CSVs (3 cities x 16 years)
models/         trained Q-tables (.npz, ~20 KB each)
results/        evaluation CSVs, leaf_metrics.json and figures used in the report and slides
docs/           report (.docx), slides (.pptx, with speaker notes), VIVA_PREP.md,
                and build_report.py / build_slides.py to regenerate them from results/
```

## Extending it

* **New city:** add it to `CITIES` in `config.py`, then retrain.
* **New crop:** add a `Crop(...)` entry with FAO-56 values, then run `python -m agrisense.train <Crop>`.
* **Different priorities:** change `RewardConfig` (for example, raise `water_cost` for water-scarce regions).
* **A different agent (e.g. DQN):** the environment returns continuous observations, so any agent
  with an `act(obs) -> mm` method plugs into `simulate.py`.
* **More diseases or crops:** add PlantVillage (or field) classes in `leaf/`, retrain and re-export the ONNX model.

## Data and references

* Weather: Open-Meteo Historical Weather API (ERA5 reanalysis). <https://open-meteo.com>
* Leaf images: PlantVillage dataset. Hughes, D. P., Salathé, M. (2015). arXiv:1511.08060.
* Allen, R. G., Pereira, L. S., Raes, D., Smith, M. (1998). *Crop evapotranspiration*, FAO Irrigation and Drainage Paper 56.
* Doorenbos, J., Kassam, A. H. (1979). *Yield response to water*, FAO Irrigation and Drainage Paper 33.
* Sutton, R. S., Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press.
* Mohanty, S. P., Hughes, D. P., Salathé, M. (2016). Using deep learning for image-based plant disease detection. *Frontiers in Plant Science*, 7, 1419.
* Sandler, M., et al. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. CVPR.
* Zhou, B., et al. (2016). Learning deep features for discriminative localization (CAM). CVPR.
