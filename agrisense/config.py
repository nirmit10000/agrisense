"""All tunable settings in one place: cities, crops, soil, reward, training.

To extend the project (new city, new crop, new soil) you only edit this file.
Crop numbers come from FAO-56 (Allen et al., 1998) Tables 11, 12, 22 and
FAO-33 (Doorenbos & Kassam, 1979) for the yield-response factor Ky.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
WEB_DATA = ROOT / "web" / "data" / "irrigation.json"   # read by the website (index.html)


def model_path(crop_name: str) -> Path:
    """Where the trained Q-table of a crop is saved."""
    return MODEL_DIR / f"q_{crop_name.lower()}.npz"


# --------------------------------------------------------------------------
# Locations - chosen to give three very different climates (max variance)
# --------------------------------------------------------------------------
CITIES = {
    "Pune": dict(lat=18.52, lon=73.86,
                 climate="Semi-arid tropical; heavy Jun-Sep monsoon, dry winter/summer"),
    "Jodhpur": dict(lat=26.24, lon=73.02,
                    climate="Arid (Thar desert); scanty, erratic rain, very high evaporation"),
    "Ludhiana": dict(lat=30.90, lon=75.85,
                     climate="Sub-tropical (Punjab breadbasket); monsoon + winter rain, cold winters"),
}
WEATHER_START = "2010-01-01"
WEATHER_END = "2025-12-31"
TRAIN_YEARS = range(2010, 2021)   # seasons starting 2010-2020 are used for training
TEST_YEARS = range(2021, 2025)    # seasons starting 2021-2024 are never seen in training


# --------------------------------------------------------------------------
# Crops (FAO-56 single crop coefficient method)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Crop:
    name: str
    stage_days: tuple       # (initial, development, mid-season, late-season)
    kc_ini: float
    kc_mid: float
    kc_end: float
    root_depth_m: float     # effective root zone depth Zr
    p: float                # fraction of TAW that can be depleted before stress (RAW = p*TAW)
    ky: float               # yield response factor (FAO-33)
    max_yield_t_ha: float   # yield with no water stress
    seasons: dict = field(default_factory=dict)  # season name -> sowing date "MM-DD"

    @property
    def season_length(self) -> int:
        return sum(self.stage_days)

    def kc(self, day: int) -> float:
        """Crop coefficient on a given day after sowing (piecewise linear FAO curve)."""
        l_ini, l_dev, l_mid, l_late = self.stage_days
        if day < l_ini:
            return self.kc_ini
        if day < l_ini + l_dev:
            return self.kc_ini + (day - l_ini) / l_dev * (self.kc_mid - self.kc_ini)
        if day < l_ini + l_dev + l_mid:
            return self.kc_mid
        return self.kc_mid + (day - l_ini - l_dev - l_mid) / l_late * (self.kc_end - self.kc_mid)

    def stage(self, day: int) -> int:
        """0 = initial, 1 = development, 2 = mid-season, 3 = late season."""
        edge = 0
        for i, length in enumerate(self.stage_days):
            edge += length
            if day < edge:
                return i
        return 3


CROPS = {
    "Maize": Crop("Maize", (20, 35, 40, 30), 0.30, 1.20, 0.60, 0.8, 0.55, 1.25, 8.0,
                  seasons={"Kharif (monsoon)": "06-15", "Zaid (summer)": "02-15"}),
    "Wheat": Crop("Wheat", (20, 25, 60, 30), 0.30, 1.15, 0.40, 0.8, 0.55, 1.15, 5.0,
                  seasons={"Rabi (winter)": "11-10"}),
    "Tomato": Crop("Tomato", (30, 40, 45, 30), 0.60, 1.15, 0.80, 0.6, 0.40, 1.05, 60.0,
                   seasons={"Rabi (winter)": "10-15", "Zaid (summer)": "02-01"}),
}
STAGE_NAMES = ["Initial", "Development", "Mid-season", "Late season"]


# --------------------------------------------------------------------------
# Soil
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Soil:
    name: str
    theta_fc: float   # volumetric water content at field capacity
    theta_wp: float   # volumetric water content at wilting point

    def taw(self, root_depth_m: float) -> float:
        """Total available water in the root zone, mm (FAO-56 eq. 82)."""
        return 1000 * (self.theta_fc - self.theta_wp) * root_depth_m


SOILS = {
    "Sandy loam": Soil("Sandy loam", 0.22, 0.10),
    "Loam": Soil("Loam", 0.30, 0.15),
}
DEFAULT_SOIL = "Sandy loam"


# --------------------------------------------------------------------------
# Environment, reward and training
# --------------------------------------------------------------------------
IRRIGATION_ACTIONS_MM = (0, 10, 20, 30, 40)   # what the agent may apply on a day
FORECAST_DAYS = 3                              # rain forecast horizon (today + next 2 days)
FORECAST_ERROR = 0.25                          # forecasts are off by ~25% (random)


@dataclass(frozen=True)
class RewardConfig:
    water_cost: float = 1.0    # penalty per mm of irrigation applied
    stress_cost: float = 10.0  # penalty per mm of crop water deficit (ETc - ETa)
    waste_cost: float = 1.0    # extra penalty per mm of irrigation lost to deep drainage
    event_cost: float = 10.0   # fixed penalty per irrigation event (pump energy, labour)


@dataclass(frozen=True)
class TrainConfig:
    episodes: int = 50000
    alpha_start: float = 0.1   # learning rate, decays linearly (like epsilon) so Q-values settle
    alpha_end: float = 0.01
    gamma: float = 0.95        # discount factor
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_frac: float = 0.7  # fraction of episodes over which epsilon decays linearly
    seed: int = 42
