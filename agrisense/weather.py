"""Download and cache daily weather from the free Open-Meteo archive API (no key needed).

Each city is stored as a small CSV in data/ (~180 KB), so everything else runs offline.
The CSVs used for the reported results are included; `python -m agrisense.weather` re-downloads them.
Columns: date, rain_mm, et0_mm (FAO Penman-Monteith reference ET), tmax_c, tmin_c
"""
import pandas as pd
import requests

from .config import CITIES, DATA_DIR, WEATHER_END, WEATHER_START

API = "https://archive-api.open-meteo.com/v1/archive"


def _cache_path(city: str):
    return DATA_DIR / f"weather_{city.lower()}.csv"


def download(city: str) -> pd.DataFrame:
    c = CITIES[city]
    params = dict(latitude=c["lat"], longitude=c["lon"],
                  start_date=WEATHER_START, end_date=WEATHER_END,
                  daily="precipitation_sum,et0_fao_evapotranspiration,temperature_2m_max,temperature_2m_min",
                  timezone="Asia/Kolkata")
    r = requests.get(API, params=params, timeout=60)
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame({
        "date": pd.to_datetime(d["time"]),
        "rain_mm": d["precipitation_sum"],
        "et0_mm": d["et0_fao_evapotranspiration"],
        "tmax_c": d["temperature_2m_max"],
        "tmin_c": d["temperature_2m_min"],
    })
    # The archive has occasional gaps; fill them by interpolation (rain gaps -> 0)
    df["rain_mm"] = df["rain_mm"].fillna(0.0)
    df = df.set_index("date").interpolate().reset_index()
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(_cache_path(city), index=False)
    return df


def load(city: str) -> pd.DataFrame:
    """Load cached weather for a city, downloading it the first time."""
    path = _cache_path(city)
    if not path.exists():
        return download(city)
    return pd.read_csv(path, parse_dates=["date"])


def load_all() -> dict:
    return {city: load(city) for city in CITIES}


if __name__ == "__main__":
    for name in CITIES:
        w = download(name)
        print(f"{name}: {len(w)} days, mean annual rain "
              f"{w.groupby(w.date.dt.year).rain_mm.sum().mean():.0f} mm")
