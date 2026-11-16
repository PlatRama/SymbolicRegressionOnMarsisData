from __future__ import annotations

import numpy as np
import pandas as pd

from config import AU_KM, C_LIGHT, CSV_PATH, DATA_CACHE, ORBIT, RESULTS_DIR, SECONDS_PER_YEAR


def load(columns=None):
    if not DATA_CACHE.exists():
        df = pd.read_csv(CSV_PATH, sep=";", engine="pyarrow")
        df.columns = [c.removeprefix("FM_data_") for c in df.columns]
        DATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(DATA_CACHE)
    return pd.read_parquet(DATA_CACHE, columns=columns)


# name -> (unit, meaning)
FEATURES = {
    "sin_ls": ("-", "Martian season; sin/cos avoid the jump at Ls = 360°"),
    "cos_ls": ("-", "Martian season"),
    "altitude_km": ("km", "spacecraft altitude"),
    "log10_altitude_km": ("log10 km", "radar law: specular echo P ~ 1/h^2, i.e. -20 log10(h) dB"),
    "freq_mhz": ("MHz", "radar frequency"),
    "wavelength_m": ("m", "wavelength c / f"),
    "roughness_over_lambda": ("-", "roughness relative to the wavelength (Rayleigh criterion)"),
    "is_day": ("bool", "Sun above the horizon at the surface"),
    "cos_sza": ("-", "cosine of the solar zenith angle; negative at night"),
    "msd_au": ("AU", "Mars-Sun distance"),
    "f107_at_mars": ("sfu", "F10.7 solar flux scaled to Mars: F10.7 / d^2"),
    "dipole_dev": ("deg", "deviation of the dipole tilt from 90°"),
    "year": ("year", "measurement date (description only, never an input)"),
    "snr": ("dB", "peak - median power; derived from the target, never an input"),
}

LEAKAGE = ["snr", "median_corrected_echo_power", "peak_distorted_echo_power", "peak_simulated_echo_power"]


def add_features(df):
    df = df.copy()
    ls = np.radians(df["solar_longitude"])
    df["sin_ls"] = np.sin(ls)
    df["cos_ls"] = np.cos(ls)

    df["altitude_km"] = df["altitude"] / 1e3
    df["log10_altitude_km"] = np.log10(df["altitude_km"])

    df["freq_mhz"] = (df["frequency"] / 1e6).round(1)
    df["wavelength_m"] = C_LIGHT / df["frequency"]
    df["roughness_over_lambda"] = df["roughness"] / df["wavelength_m"]

    df["is_day"] = df["Sun_elevation_angle"] > 0
    df["cos_sza"] = np.sin(np.radians(df["Sun_elevation_angle"]))
    df["msd_au"] = df["Mars_Sun_distance"] / AU_KM
    df["f107_at_mars"] = df["F10_7_index"] / df["msd_au"] ** 2

    df["dipole_dev"] = (df["dipole_tilt"] - 90).abs()

    df["year"] = 2000 + df["ephemeris_time"] / SECONDS_PER_YEAR  # ephemeris time: seconds since J2000
    df["snr"] = df["peak_corrected_echo_power"] - df["median_corrected_echo_power"]
    return df


def load_with_features(columns=None):
    return add_features(load(columns=columns))


def check_inputs(features):
    bad = [f for f in features if f in LEAKAGE]
    if bad:
        raise ValueError(f"these columns are derived from the target and cannot be inputs: {bad}")
    return features


def describe_features(names=None):
    names = names or list(FEATURES)
    return "\n".join(f"  {n:22s} [{FEATURES[n][0]}]  {FEATURES[n][1]}" for n in names if n in FEATURES)


PERIODS = ("all", "night", "day", "lit")


def select(df, freqs="all", period="all", sun_margin=0.0, cos_sza_min=-0.2):
    """night: Sun < -sun_margin; day: Sun > sun_margin; lit: cos_sza > cos_sza_min."""
    if period not in PERIODS:
        raise ValueError(f"period must be one of {PERIODS}, not {period!r}")
    mask = pd.Series(True, index=df.index)
    if freqs != "all":
        mask &= df["freq_mhz"].isin([float(f) for f in freqs])
    sea = df["Sun_elevation_angle"]
    if period == "night":
        mask &= sea < -sun_margin
    elif period == "day":
        mask &= sea > sun_margin
    elif period == "lit":
        mask &= df["cos_sza"] > cos_sza_min
    return df[mask]


def anomalous_orbits_file(experiment_name):
    return RESULTS_DIR / experiment_name / "anomalous_orbits.csv"


def load_anomalous_orbits(experiment_name):
    path = anomalous_orbits_file(experiment_name)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run "
                                f"python src/run_experiments.py --exp {experiment_name} --step anomalies")
    return pd.read_csv(path)[ORBIT].to_numpy()


def derived_target_file(experiment_name, target):
    return RESULTS_DIR / experiment_name / f"{target}.parquet"


def subset(df, cfg):
    """Rows of an experiment. A target that is not a dataset column is read from
    the file written by the step that computes it."""
    sub = select(df, freqs=cfg.freqs, period=cfg.period, sun_margin=cfg.sun_margin, cos_sza_min=cfg.cos_sza_min)
    if cfg.exclude_anomalies_of:
        sub = sub[~sub[ORBIT].isin(load_anomalous_orbits(cfg.exclude_anomalies_of))]
    if cfg.target not in sub:
        path = derived_target_file(cfg.name, cfg.target)
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run "
                                    f"python src/run_experiments.py --exp {cfg.name} --step {cfg.target}")
        sub = sub.join(pd.read_parquet(path)[[cfg.target]], how="inner")
    return sub


def binned_median(x, y, bins, groups, min_count=500):
    """Median of y in bins of x for every group; bins with fewer than min_count rows are dropped."""
    d = pd.DataFrame({"x": np.asarray(x), "y": np.asarray(y), "group": np.asarray(groups)})
    d["bin"] = pd.cut(d["x"], bins)
    t = d.groupby(["group", "bin"], observed=True)["y"].agg(median="median", n="size").reset_index()
    t.insert(1, "x", [iv.mid for iv in t["bin"]])
    return t[t["n"] >= min_count].drop(columns="bin")


def count_by(df, column="freq_mhz"):
    return {float(k): int(v) for k, v in df[column].value_counts().sort_index().items()}


FOLD_SCHEMES = ("chronological", "random_orbits")


def make_folds(df, n_folds=10, scheme="chronological", seed=0):
    """(train index, test index) pairs. Orbits are never split between train and test."""
    orbits = np.sort(df[ORBIT].unique())
    if scheme == "random_orbits":
        orbits = np.random.default_rng(seed).permutation(orbits)
    elif scheme != "chronological":
        raise ValueError(f"scheme must be one of {FOLD_SCHEMES}, not {scheme!r}")

    folds = []
    for test_orbits in np.array_split(orbits, n_folds):
        is_test = df[ORBIT].isin(test_orbits).to_numpy()
        folds.append((df.index[~is_test], df.index[is_test]))
    return folds


def folds_for(df, cfg):
    return make_folds(df, n_folds=cfg.n_folds, scheme=cfg.fold_scheme)
