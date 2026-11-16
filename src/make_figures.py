from __future__ import annotations

import argparse
import calendar
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import pandas as pd

from config import EXPERIMENTS, RESULTS_DIR, get_experiment
from plot_results import BASELINE_STYLE, FREQ_COLORS, SERIES_COLORS, save

STEP_NUMBERS = {"explore": 1, "search": 2, "validate": 3, "residuals": 4, "anomalies": 5,
                "attenuation": 6, "exponents": 7}


def name(exp, step):
    return f"{STEP_NUMBERS[step]:02d}_{exp}_{step}.png"


def read(exp, filename):
    path = RESULTS_DIR / exp / filename
    return pd.read_csv(path) if path.exists() else None


def month_year(y):
    return f"{calendar.month_abbr[int((y % 1) * 12) + 1]} {int(y)}"


def fig_explore(exp):
    orbits, folds = read(exp, "explore_orbits.csv"), read(exp, "fold.csv")
    if orbits is None:
        return
    orbits = orbits.sort_values("year").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    sc = ax.scatter(orbits["year"], orbits["sun"], c=orbits["f107"], cmap="viridis", s=12)
    fig.colorbar(sc, ax=ax, label="F10.7 (solar activity)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xlabel="year", ylabel="mean Sun elevation of the orbit [°]",
           title=f"{exp}: one point per orbit over the years")
    save(fig, f"{STEP_NUMBERS['explore']:02d}_{exp}_explore_years.png")

    starts = list(np.searchsorted(orbits["year"].to_numpy(), folds["start_year"].to_numpy())) + [len(orbits)]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, (a, b) in enumerate(zip(starts[:-1], starts[1:])):
        if i % 2 == 0:
            ax.axvspan(a - 0.5, b - 0.5, color="grey", alpha=0.12, lw=0)
        ax.text((a + b - 1) / 2, 1.01, f"fold {i}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8)
    sc = ax.scatter(orbits.index, orbits["sun"], c=orbits["f107"], cmap="viridis", s=12, zorder=3)
    fig.colorbar(sc, ax=ax, label="F10.7 (solar activity)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(starts[:-1], [month_year(orbits["year"][s]) for s in starts[:-1]], fontsize=8)
    ax.set_xlim(-1, len(orbits))
    ax.grid(axis="x", visible=False)
    ax.set(xlabel="orbits in time order (label = date of the first orbit of each fold)",
           ylabel="mean Sun elevation of the orbit [°]")
    ax.set_title(f"{exp}: one point per orbit; grey/white bands = folds", pad=18)
    save(fig, name(exp, "explore"))


def fig_search(exp):
    pareto, baselines = read(exp, "pareto.csv"), read(exp, "search_baselines.csv")
    if pareto is None:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for (seed, p), color in zip(pareto.groupby("seed"), SERIES_COLORS):
        p = p.sort_values("complexity")
        ax.plot(p["complexity"], p["mae_train"], "o--", color=color, ms=3, lw=1, alpha=0.6)
        ax.plot(p["complexity"], p["mae_test"], "o-", color=color, ms=5, label=f"seed {seed}")
    for _, b in baselines.iterrows():
        color, ls = BASELINE_STYLE[b["model"]]
        ax.axhline(b["mae_test"], color=color, ls=ls, lw=1.3, label=f"{b['model']} (test)")
    ax.plot([], [], "k-", label="PySR test (solid)")
    ax.plot([], [], "k--", alpha=0.6, label="PySR train (dashed)")
    ax.set(xlabel="formula complexity", ylabel="MAE [dB]", title=f"{exp}: PySR Pareto front")
    ax.legend(fontsize=8, ncol=2)
    save(fig, name(exp, "search"))


def fig_validate(exp):
    res = read(exp, "validate.csv")
    if res is None:
        return
    cfg = get_experiment(exp)
    order = cfg.baselines + cfg.formulas
    per_fold = res.pivot(index="fold", columns="model", values="mae")[order]
    mean = per_fold.mean()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = iter(SERIES_COLORS)
    for model in order:
        color, ls = BASELINE_STYLE.get(model) or (next(colors), "-")
        ax.plot(per_fold.index, per_fold[model], marker="o", ms=4, ls=ls, color=color,
                label=f"{model} ({mean[model]:.2f})")
    ax.set(xlabel="fold (blocks of orbits in time order)", ylabel="MAE [dB]", xticks=per_fold.index,
           title=f"{exp}: error per fold (mean in brackets)")
    ax.legend(fontsize=8)
    save(fig, name(exp, "validate"))


def fig_residuals(exp):
    hist, by_sun, orbit = read(exp, "residuals_hist.csv"), read(exp, "residuals_by_sun.csv"), \
        read(exp, "residuals_by_orbit.csv")
    if hist is None:
        return
    cfg = get_experiment(exp)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].bar(hist["left"], hist["rows"], width=hist["right"] - hist["left"], align="edge", color="tab:blue")
    ax[0].set(xlabel="residual [dB]", ylabel="rows", title="Residual distribution")
    ax[0].text(0.02, 0.97, f"residuals above {hist['right'].max():g} dB\nare counted in the last bar",
               transform=ax[0].transAxes, va="top", fontsize=8, color="0.3")
    for f, g in by_sun.groupby("freq_mhz"):
        sparse = len(g) < 0.5 * by_sun["cos_sza"].nunique()
        ax[1].plot(g["cos_sza"], g["median"], "o" if sparse else "o-", ms=4, color=FREQ_COLORS.get(f),
                   label=f"{f:g} MHz" + (" (few bins, points only)" if sparse else ""))
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].axvline(0, color="k", lw=0.8, ls=":")
    ax[1].legend()
    ax[1].set(xlabel="cos(solar zenith angle)   ← night | day →", ylabel="median residual [dB]",
              title="Residual vs Sun, by frequency")
    sc = ax[2].scatter(orbit["year"], orbit["bias_dB"], c=orbit["f107"], cmap="viridis", s=10)
    fig.colorbar(sc, ax=ax[2], label="F10.7")
    ax[2].axhline(cfg.anomaly_threshold, color="k", ls="--", lw=0.8,
                  label=f"anomaly threshold ({cfg.anomaly_threshold:g} dB)")
    ax[2].legend(fontsize=8, loc="lower left")
    ax[2].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[2].set(xlabel="year", ylabel="mean residual of the orbit [dB]", title="Mean residual per orbit")
    fig.suptitle(f"{exp}: residuals of {cfg.residual_model}")
    save(fig, name(exp, "residuals"))


def fig_anomalies(exp):
    orbits = read(exp, "orbit_summary.csv")
    if orbits is None:
        return
    cfg = get_experiment(exp)
    rho = json.loads((RESULTS_DIR / exp / "summary.json").read_text())["anomalies"]["spearman_residual_noise"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for flag, color, label in [(False, "tab:blue", "normal"), (True, "tab:red", "anomalous")]:
        s = orbits[orbits["anomalous"] == flag]
        ax[0].scatter(s["bias"], s["median_corrected_echo_power"], s=12, c=color, label=f"{label} ({len(s)})")
        ax[1].scatter(s["year"], s["bias"], s=12, c=color, label=f"{label} ({len(s)})")
    threshold = f"anomaly threshold ({cfg.anomaly_threshold:g} dB)"
    ax[0].axvline(cfg.anomaly_threshold, color="k", ls="--", lw=0.8, label=threshold)
    ax[1].axhline(cfg.anomaly_threshold, color="k", ls="--", lw=0.8, label=threshold)
    ax[0].legend()
    ax[1].legend(fontsize=8, loc="lower left")
    ax[0].set(xlabel="mean residual of the orbit [dB]", ylabel="mean noise floor of the orbit [dB]",
              title=f"Do echo and noise floor drop together? (Spearman {rho:.2f})")
    ax[1].set(xlabel="year", ylabel="mean residual of the orbit [dB]", title="Anomalous orbits in time")
    fig.suptitle(f"{exp}: anomalous orbits with respect to {cfg.anomaly_formula}")
    save(fig, name(exp, "anomalies"))


def fig_attenuation(exp):
    t = read(exp, "attenuation_by_cos_sza.csv")
    if t is None:
        return
    cfg = get_experiment(exp)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for f, g in t.groupby("freq_mhz"):
        sparse = len(g) < 0.5 * t["cos_sza"].nunique()
        ax.plot(g["cos_sza"], g["median"], "o" if sparse else "o-", ms=4, color=FREQ_COLORS.get(f),
                label=f"{f:g} MHz" + (" (few bins, points only)" if sparse else ""))
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.8, ls=":")
    ax.axvline(cfg.cos_sza_min, color="grey", lw=0.8, ls="--",
               label=f"start of the lit rows (cos χ = {cfg.cos_sza_min:g})")
    ax.legend()
    ax.set(xlabel="cos(solar zenith angle)   ← night | day →", ylabel="median attenuation [dB]",
           title=f"Signal lost with respect to the night law ({cfg.reference_formula})")
    save(fig, name(exp, "attenuation"))


def fig_exponents(exp):
    boot, table = read(exp, "exponents_bootstrap.csv"), read(exp, "exponents.csv")
    if boot is None:
        return
    cfg = get_experiment(exp)
    table = table.set_index("parameter")
    exps = [p for p in boot.columns if p in cfg.theory]
    fig, axes = plt.subplots(1, len(exps), figsize=(4.5 * len(exps), 3.8))
    for ax, p in zip(axes, exps):
        ax.hist(boot[p], bins=15, color="tab:blue", alpha=0.7)
        ax.axvspan(table.loc[p, "ci95_low"], table.loc[p, "ci95_high"], color="tab:blue", alpha=0.12, label="95% CI")
        ax.axvline(table.loc[p, "estimate"], color="tab:blue", lw=2, label=f"estimate {table.loc[p, 'estimate']:.2f}")
        ax.axvline(cfg.theory[p], color="tab:red", ls="--", lw=2, label=f"theory {cfg.theory[p]:g}")
        ax.set(xlabel=p, ylabel="resamplings", title=f"exponent {p}")
        ax.legend(fontsize=8)
    fig.suptitle(f"{exp}: exponents of {cfg.exponent_formula}, orbit bootstrap ({len(boot)})")
    save(fig, name(exp, "exponents"))


FIGURES = [fig_explore, fig_search, fig_validate, fig_residuals, fig_anomalies, fig_attenuation, fig_exponents]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    args = parser.parse_args()
    for exp in args.exp:
        for draw in FIGURES:
            draw(exp)


if __name__ == "__main__":
    main()
