# AgriSense: smart irrigation with reinforcement learning

**EC-422 Artificial Intelligence, course project.** Dushyant Kumar (23145030), Nirmit Patil (23145053)
Theme: Agriculture: intelligent irrigation, crop monitoring and resource optimisation.

- **Website:** <https://nirmit10000.github.io/agrisense/>
- **Presentation (Canva):** <https://www.canva.com/design/DAHaf0-Y1zI/9xi2UGtj7G3lCuj4Jpl6Uw/view>
- **Report:** [AgriSense_Report.pdf](AgriSense_Report.pdf)

A tabular Q-learning agent decides every day how much to irrigate a field (0, 10, 20, 30 or 40 mm).
It learns by practising 50,000 seasons per crop in a field simulator (FAO-56 soil-water balance) driven by
real daily weather from three Indian climates (Pune, Jodhpur, Ludhiana). It is trained on seasons from
2010-2020 and tested on 60 seasons from 2021-2024 that it never saw.

| Policy (average of 60 unseen test seasons) | Water per season | Yield (% of max) | Crop per m³ of water |
|---|---|---|---|
| **Q-learning agent (RL)** | **435 mm** | 98.1 % | **6.30 kg** |
| Sensor threshold rule (FAO) | 460 mm | 100.0 % | 6.13 kg |
| Fixed schedule (40 mm every 7 days) | 784 mm | 93.7 % | 3.33 kg |
| Rainfed (no irrigation) | 0 mm | 31.8 % | - |

The agent uses 45% less water than the fixed schedule and 5.5% less than the sensor rule. It keeps 98% of
maximum yield.

## Contents

| Path | What it is |
|---|---|
| `agrisense/config.py` | All settings: cities, crops (FAO-56 values), soil, reward weights, training hyper-parameters, file paths |
| `agrisense/weather.py` | Downloads the weather dataset from the Open-Meteo archive API (ERA5) |
| `agrisense/env.py` | The field simulator (the RL environment): daily soil-water balance and FAO-33 yield |
| `agrisense/agents.py` | The three baseline policies, the state encoder and the Q-learning agent |
| `agrisense/train.py` | Trains one Q-learning agent per crop |
| `agrisense/evaluate.py` | Tests all policies on 2021-2024, draws the figures and writes the website data |
| `data/` | The dataset: daily weather 2010-2025 for the three cities (3 CSV files) |
| `models/` | The trained Q-tables (one per crop) |
| `results/` | Test results (CSV) and figures |
| `index.html`, `web/` | The website (static HTML, CSS and JavaScript, with ECharts for the charts) |
| `AgriSense_Report.pdf` | Project report |

## Run it

Python 3.10 or newer. Run every command from this folder.

```bash
pip install -r requirements.txt
python -m agrisense.train      # train the 3 agents (about 3.5 min on a laptop CPU) -> models/
python -m agrisense.evaluate   # test results and figures -> results/, website data -> web/data/irrigation.json
python -m http.server          # then open http://localhost:8000 to view the website
```

The trained models, results and website data are already included, so you can skip straight to the last
command. Opening `index.html` directly from disk does not work, because the browser blocks loading the data
file from `file://`.

**Reproducibility.** Training uses a fixed random seed (42). Running `train` and then `evaluate` regenerates
every file in `models/`, `results/` and `web/data/` byte for byte. This was checked with Python 3.11,
NumPy 2.2, pandas 2.3 and Matplotlib 3.10.

**Dataset.** `data/` holds the exact weather used for the results. `python -m agrisense.weather` downloads
it again from Open-Meteo (free, no key). The archive is occasionally revised, so a fresh download can
differ slightly.

The website is hosted on GitHub Pages straight from this repository (`index.html` at the root).
