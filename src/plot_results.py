from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PLOTS_DIR

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
FREQ_COLORS = {1.8: "tab:purple", 3.0: "tab:green", 4.0: "tab:blue", 5.0: "tab:orange"}
# formulas and PySR seeds; green, red and black are kept for the baselines
SERIES_COLORS = ["tab:blue", "tab:orange", "tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan"]
BASELINE_STYLE = {"constant": ("k", ":"), "linear": ("tab:green", ":"), "GBM": ("tab:red", ":")}


def save(fig, filename, log=print):
    PLOTS_DIR.mkdir(exist_ok=True)
    path = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    log(f"figure: plots/{filename}")
    return path
