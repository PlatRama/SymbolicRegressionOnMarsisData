from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = Path(r"D:\MARSIS_historical_dataset.csv")
CSV_PATH = ROOT / "data" / "MARSIS_historical_dataset.csv"
DATA_CACHE = ROOT / "data" / "marsis.parquet"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = ROOT / "plots"
FORMULAS_FILE = ROOT / "formulas.toml"

TARGET = "peak_corrected_echo_power"
ORBIT = "orbit_number"
FREQS_MHZ = (1.8, 3.0, 4.0, 5.0)

C_LIGHT = 299_792_458.0
AU_KM = 149_597_870.7
SECONDS_PER_YEAR = 365.25 * 86400

NIGHT_FEATURES = ["log10_altitude_km", "roughness_over_lambda", "slope", "f107_at_mars",
                  "cos_sza", "sin_ls", "cos_ls", "dipole_dev"]


@dataclass
class ExperimentConfig:
    name: str
    description: str
    target: str = TARGET

    # data
    freqs: list | str = "all"            # "all" or a list of MHz, e.g. [4.0]
    period: str = "all"                  # "all" | "night" | "day" | "lit"
    sun_margin: float = 0.0              # degrees excluded around the horizon for "night"/"day"
    cos_sza_min: float = -0.2            # "lit": ionosphere illuminated, twilight included
    exclude_anomalies_of: str | None = None
    features: list = field(default_factory=lambda: list(NIGHT_FEATURES))

    # folds
    n_folds: int = 10
    fold_scheme: str = "chronological"   # "chronological" | "random_orbits"

    # PySR
    search_test_fold: int = 9
    pysr_sample: int = 10_000
    pysr_balance_by: str | None = None
    pysr_iterations: int = 200
    pysr_maxsize: int = 25
    pysr_seeds: tuple = (0,)
    pysr_timeout: int = 0
    pysr_binary: list = field(default_factory=lambda: ["+", "-", "*", "/"])
    pysr_unary: list = field(default_factory=lambda: ["log10", "sqrt", "square", "exp"])
    pysr_nested: dict = field(default_factory=lambda: {"log10": {"log10": 0, "exp": 0},
                                                       "exp": {"exp": 0, "log10": 0}})
    pysr_constraints: dict | None = None

    # validation
    formulas: list = field(default_factory=list)
    baselines: list = field(default_factory=lambda: ["constant", "linear", "GBM"])
    train_sample: int = 500_000
    group_by: str | None = None

    residual_model: str | None = None

    # anomalous orbits
    anomaly_formula: str | None = None
    anomaly_threshold: float = -8.0
    anomaly_max_iter: int = 10
    anomaly_thresholds: tuple = (-6, -8, -10, -12)

    # daytime attenuation
    reference_formula: str | None = None
    reference_experiment: str | None = None
    exponent_formula: str | None = None
    bootstrap: int = 40
    bootstrap_sample: int = 150_000
    theory: dict = field(default_factory=dict)


_all_data = ExperimentConfig(
    name="all_data",
    description="Whole dataset: is there a single function that explains all the data?",
    features=NIGHT_FEATURES + ["freq_mhz"],
    pysr_sample=12_000, pysr_balance_by="freq_mhz",
    group_by="freq_mhz",
    residual_model="all_data_c7",
)

_night_4mhz = ExperimentConfig(
    name="night_4mhz",
    description="Night at 4 MHz (all orbits): radar law and anomalous orbits",
    freqs=[4.0], period="night", sun_margin=10,
    pysr_iterations=1000, pysr_seeds=(0, 1, 2),
    formulas=["night_4mhz_c5", "night_4mhz_c6", "night_4mhz_c7", "night_4mhz_c10", "radar_4mhz_f107"],
    residual_model="radar_4mhz_f107",
    anomaly_formula="radar_4mhz_f107",
)

_night_allfreq = ExperimentConfig(
    name="night_allfreq",
    description="Night, all frequencies, no anomalous orbits: radar law and frequency gain",
    period="night", sun_margin=10, exclude_anomalies_of="night_4mhz",
    features=NIGHT_FEATURES + ["freq_mhz"],
    pysr_sample=12_000, pysr_balance_by="freq_mhz",
    pysr_iterations=1000, pysr_seeds=(0, 1, 2),
    formulas=["night_allfreq_c14", "night_allfreq_offset", "radar_night_resonance", "radar_night"],
    group_by="freq_mhz",
)

_attenuation = ExperimentConfig(
    name="attenuation",
    description="Day: ionospheric attenuation law (echo expected at night - measured echo)",
    target="attenuation",
    period="lit", cos_sza_min=-0.2, exclude_anomalies_of="night_4mhz",
    features=["cos_sza", "f107_at_mars", "freq_mhz"],
    pysr_sample=12_000, pysr_balance_by="freq_mhz", pysr_iterations=1000, pysr_seeds=(0, 1, 2),
    pysr_maxsize=20,
    pysr_binary=["+", "-", "*", "/", "^"],
    pysr_unary=["sqrt", "square", "exp", "log10"],
    pysr_constraints={"^": (-1, 1)},
    pysr_nested={"^": {"^": 0, "exp": 0}, "exp": {"exp": 0, "^": 0, "log10": 0},
                 "log10": {"log10": 0, "exp": 0}},
    formulas=["attenuation_c7", "attenuation_pysr", "attenuation_free", "attenuation_chapman"],
    group_by="freq_mhz",
    reference_formula="radar_night_resonance", reference_experiment="night_allfreq",
    exponent_formula="attenuation_free", theory={"m": 0.5, "p": 0.5, "n": 2.0},
)

_combined = ExperimentConfig(
    name="combined",
    description="Whole dataset without anomalous orbits: night law - daytime attenuation",
    exclude_anomalies_of="night_4mhz",
    features=NIGHT_FEATURES + ["freq_mhz"],
    formulas=["all_data_c7", "radar_night_resonance", "combined", "combined_free"],
    group_by="freq_mhz",
    residual_model="combined",
)

EXPERIMENTS = {
    "all_data": _all_data,
    "all_data_long": replace(
        _all_data, name="all_data_long", pysr_iterations=2000, pysr_seeds=(0, 1, 2),
        description="Whole dataset, long search (2000 iterations, 3 seeds): do the formulas change?"),
    "night_4mhz": _night_4mhz,
    "night_4mhz_clean": replace(
        _night_4mhz, name="night_4mhz_clean", exclude_anomalies_of="night_4mhz",
        description="Night at 4 MHz without anomalous orbits"),
    "night_allfreq": _night_allfreq,
    "attenuation": _attenuation,
    "combined": _combined,
    "combined_all": replace(
        _combined, name="combined_all", exclude_anomalies_of=None,
        description="Whole dataset including anomalous orbits"),
}


def get_experiment(name: str) -> ExperimentConfig:
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {name!r}. Available: {list(EXPERIMENTS)}")
    return EXPERIMENTS[name]
