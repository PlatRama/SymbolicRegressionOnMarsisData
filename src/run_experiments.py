from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

import data
from anomalies import flag_anomalous_orbits, group_episodes, orbit_summary, threshold_sensitivity
from config import EXPERIMENTS, ORBIT, RESULTS_DIR, TARGET, get_experiment
from evaluate import BASELINES, cross_validate, mae, mae_by_group, model_factory, out_of_fold_predictions
from formulas import FormulaModel, load_formula
from results_writer import ResultsWriter
from search import run_search


def fmt_params(params):
    return ", ".join(f"{k} = {v:.3f}" for k, v in params.items())


# 01 explore

def step_explore(cfg, out):
    df = data.load_with_features()
    sub = data.subset(df, cfg)
    out.log(f"Selected rows: {len(sub):,} of {len(df):,} ({100 * len(sub) / len(df):.1f}%), "
            f"orbits: {sub[ORBIT].nunique()}")
    out.log(f"Rows per frequency: {data.count_by(sub)}")
    if cfg.target in sub:
        out.log(f"Target ({cfg.target}): mean {sub[cfg.target].mean():.2f} dB, std {sub[cfg.target].std():.2f} dB")

    folds = data.folds_for(sub, cfg)
    rows = []
    for k, (_, test_idx) in enumerate(folds):
        t = sub.loc[test_idx]
        row = {"fold": k,
               "start_year": t["year"].min(),
               "period": f"{t['year'].min():.1f}-{t['year'].max():.1f}",
               "orbits": t[ORBIT].nunique(),
               "rows": len(t),
               "pct_day": 100 * t["is_day"].mean(),
               "sun_elev": t["Sun_elevation_angle"].mean(),
               "F10.7": t["F10_7_index"].mean(),
               "altitude_km": t["altitude_km"].mean()}
        if cfg.target in t:
            row["target_dB"] = t[cfg.target].mean()
        row.update({f"rows_{f:g}MHz": n for f, n in data.count_by(t).items()})
        rows.append(row)
    fold_table = pd.DataFrame(rows).set_index("fold").fillna(0)
    out.show(fold_table, "Observing conditions in the test set of each fold", digits=1)
    out.save_table(fold_table, "fold.csv", index=True)
    out.update_summary({"rows": len(sub), "orbits": sub[ORBIT].nunique(),
                        "rows_per_frequency": data.count_by(sub)})

    orbits = sub.groupby(ORBIT).agg(year=("year", "mean"), sun=("Sun_elevation_angle", "mean"),
                                    f107=("F10_7_index", "mean"))
    out.save_table(orbits.reset_index(), "explore_orbits.csv")


# 02 search

def step_search(cfg, out):
    """PySR searches on all folds except search_test_fold; every Pareto formula is scored on the full
    train and on the unseen test fold, next to the baselines."""
    df = data.load_with_features()
    sub = data.subset(df, cfg)
    data.check_inputs(cfg.features)
    train_idx, test_idx = data.folds_for(sub, cfg)[cfg.search_test_fold]
    train, test = sub.loc[train_idx], sub.loc[test_idx]
    out.log(f"Train: {len(train):,} rows (folds other than {cfg.search_test_fold}), "
            f"test: {len(test):,} rows ({test['year'].min():.1f}-{test['year'].max():.1f})")
    out.log(f"PySR inputs: {cfg.features}")
    out.log(f"PySR: sample {cfg.pysr_sample:,} rows"
            f"{f' balanced by {cfg.pysr_balance_by}' if cfg.pysr_balance_by else ''}, "
            f"{cfg.pysr_iterations} iterations, max complexity {cfg.pysr_maxsize}, seeds {list(cfg.pysr_seeds)}")

    t0 = time.time()
    pareto = run_search(train, test, cfg, log=out.log)
    out.log(f"Search finished in {(time.time() - t0) / 60:.1f} minutes")

    rows = []
    for name in cfg.baselines:
        fit_on = train.sample(min(cfg.train_sample, len(train)), random_state=0)
        model = BASELINES[name]()
        model.fit(fit_on[cfg.features], fit_on[cfg.target])
        pred = model.predict(test[cfg.features])
        row = {"model": name, "mae_test": mae(test[cfg.target], pred)}
        if cfg.group_by:
            row.update({f"mae_test_{cfg.group_by}={g:g}": v
                        for g, v in mae_by_group(test[cfg.target], pred, test[cfg.group_by]).items()})
        rows.append(row)
    baselines = pd.DataFrame(rows)

    out.show(pareto.drop(columns="seed") if len(cfg.pysr_seeds) == 1 else pareto, "Pareto front (MAE in dB)")
    out.show(baselines, "Baselines on the same test fold")
    out.save_table(pareto, "pareto.csv")
    out.save_table(baselines, "search_baselines.csv")

    best = pareto.loc[pareto["mae_test"].idxmin()]
    out.update_summary({"search": {
        "test_fold": cfg.search_test_fold,
        "best_on_test": {"complexity": best["complexity"], "mae_test": best["mae_test"],
                         "equation": best["equation"]},
        "baseline_mae_test": dict(zip(baselines["model"], baselines["mae_test"])),
    }})


# 03 validate

def step_validate(cfg, out):
    """Formulas keep their structure; their constants are refitted on the train set of every fold."""
    df = data.load_with_features()
    sub = data.subset(df, cfg)
    data.check_inputs(cfg.features)
    folds = data.folds_for(sub, cfg)
    out.log(f"Rows: {len(sub):,}, folds: {len(folds)} ({cfg.fold_scheme})")

    results, params = [], []
    for name in cfg.baselines + cfg.formulas:
        make_model, feats = model_factory(name, cfg.features)
        out.log(f"\n{name}  (columns: {feats})")
        res, par = cross_validate(make_model, sub, feats, cfg.target, folds, train_sample=cfg.train_sample,
                                  group_by=cfg.group_by, log=out.log)
        res.insert(0, "model", name)
        results.append(res)
        if par is not None:
            params.append(par.reset_index().melt(id_vars="fold", var_name="parameter", value_name="value")
                          .assign(model=name))
    results = pd.concat(results, ignore_index=True)
    order = cfg.baselines + cfg.formulas

    per_fold = results.pivot(index="fold", columns="model", values="mae")[order]
    out.show(per_fold, "MAE per fold [dB]", digits=2)
    mean_cols = ["mae"] + [c for c in results.columns if c.startswith("mae_")]
    summary = results.groupby("model")[mean_cols].mean().loc[order]
    out.show(summary, "Mean MAE over the folds [dB]", digits=2)
    out.save_table(results, "validate.csv")

    stab = None
    if params:
        params = pd.concat(params, ignore_index=True)
        stab = params.groupby(["model", "parameter"], sort=False)["value"].agg(["mean", "std", "min", "max"])
        out.show(stab, "Formula constants fitted in the folds")
        out.save_table(params, "validate_params.csv")

    out.update_summary({"validate": {
        "mean_mae": summary["mae"].to_dict(),
        "mean_constants": ({f"{m}:{p}": v for (m, p), v in stab["mean"].items()} if stab is not None else {}),
    }})


# 04 residuals

RESIDUAL_CANDIDATES = ["cos_sza", "Sun_elevation_angle", "local_true_solar_time", "freq_mhz", "band",
                       "log10_altitude_km", "roughness_over_lambda", "slope", "dipole_dev", "monopole_tilt",
                       "F10_7_index", "f107_at_mars", "msd_au", "sin_ls", "cos_ls", "year",
                       "latitude", "longitude", "radial_magnetic_field", "tangential_magnetic_field",
                       "total_magnetic_field"]


def sun_period(sea):
    return pd.cut(sea, [-90, -10, 0, 90], labels=["night", "twilight", "day"])


def step_residuals(cfg, out):
    """Out-of-fold residuals (measured - predicted) by period, frequency, orbit and variable."""
    if not cfg.residual_model:
        raise SystemExit(f"{cfg.name}: residual_model missing in config.py")
    df = data.load_with_features()
    sub = data.subset(df, cfg).copy()
    folds = data.folds_for(sub, cfg)
    make_model, feats = model_factory(cfg.residual_model, cfg.features)
    out.log(f"Formula: {cfg.residual_model}  ({make_model().formula.expression})")

    sub["pred"] = out_of_fold_predictions(make_model, sub, feats, cfg.target, folds, train_sample=cfg.train_sample)
    sub["resid"] = sub[cfg.target] - sub["pred"]
    sub["period"] = sun_period(sub["Sun_elevation_angle"])
    r = sub["resid"]
    out.log(f"Out-of-fold residuals: MAE {r.abs().mean():.2f} dB, mean {r.mean():.2f}, median {r.median():.2f}, "
            f"skewness {r.skew():.2f}")
    out.log("Quantiles: " + ", ".join(f"{int(q * 100)}%: {v:.1f}"
                                      for q, v in r.quantile([.01, .05, .25, .5, .75, .95, .99]).items()))

    agg = dict(rows="size", bias_dB="mean", mae_dB=lambda s: s.abs().mean())
    by_period = sub.groupby("period", observed=True)["resid"].agg(**agg)
    by_group = sub.groupby(["period", "freq_mhz"], observed=True)["resid"].agg(**agg)
    out.show(by_period, "Residuals by period (bias = mean; negative = formula overestimates)", digits=2)
    out.show(by_group, "Residuals by period and frequency", digits=2)
    out.save_table(by_group.reset_index(), "residuals_by_group.csv")

    orbit = sub.groupby(ORBIT).agg(n=("resid", "size"), bias_dB=("resid", "mean"), year=("year", "mean"),
                                   f107=("F10_7_index", "mean"), sun=("Sun_elevation_angle", "mean"))
    between = sub.groupby(ORBIT)["resid"].transform("mean")
    within = r - between
    low = orbit[orbit["bias_dB"] < cfg.anomaly_threshold]
    out.log(f"\nResidual variance {r.var():.1f} = between orbits {between.var():.1f} + within orbits {within.var():.1f}")
    out.log(f"Orbits with mean residual < {cfg.anomaly_threshold} dB: {len(low)} of {len(orbit)} "
            f"({100 * low['n'].sum() / orbit['n'].sum():.1f}% of rows); at night (mean Sun < -10°): "
            f"{(low['sun'] < -10).sum()}")
    out.save_table(orbit.reset_index(), "residuals_by_orbit.csv")

    sample = sub.sample(min(300_000, len(sub)), random_state=0)
    corr = sample[RESIDUAL_CANDIDATES].corrwith(sample["resid"], method="spearman").sort_values(key=abs, ascending=False)
    out.show(corr.to_frame("spearman"), "Spearman correlation between residual and variable")

    out.update_summary({"residuals": {
        "formula": cfg.residual_model,
        "mae": r.abs().mean(),
        "by_period": {str(k): {"bias_dB": v["bias_dB"], "mae_dB": v["mae_dB"]} for k, v in by_period.iterrows()},
        "variance_between_orbits": between.var(), "variance_within_orbits": within.var(),
        "orbits_below_threshold": len(low),
        "top_correlations": corr.head(5).to_dict(),
    }})

    counts, edges = np.histogram(r.clip(-25, 15), bins=120)
    out.save_table(pd.DataFrame({"left": edges[:-1], "right": edges[1:], "rows": counts}), "residuals_hist.csv")
    out.save_table(data.binned_median(sub["cos_sza"], r, np.linspace(-0.75, 0.6, 28), sub["freq_mhz"])
                   .rename(columns={"group": "freq_mhz", "x": "cos_sza"}), "residuals_by_sun.csv")


# 05 anomalies

ANOMALY_COLUMNS = ["year", TARGET, "median_corrected_echo_power", "snr", "F10_7_index", "f107_at_mars",
                   "Sun_elevation_angle", "altitude_km", "roughness_over_lambda", "dipole_dev"]


def step_anomalies(cfg, out):
    """Iterative search of the anomalous orbits, then the cause is checked on the noise floor
    (median_corrected_echo_power), which the formula does not use: if it drops with the echo, the
    signal was absorbed in the ionosphere. Writes anomalous_orbits.csv for exclude_anomalies_of."""
    if not cfg.anomaly_formula:
        raise SystemExit(f"{cfg.name}: anomaly_formula missing in config.py")
    df = data.load_with_features()
    sub = data.subset(df, cfg)
    formula = load_formula(cfg.anomaly_formula)
    out.log(f"Reference formula: {formula.name}  ({formula.expression})")
    out.log(f"Threshold: mean residual of the orbit < {cfg.anomaly_threshold} dB")

    out.log("Iterative search:")
    anomalous, orbit_bias, model = flag_anomalous_orbits(sub, formula, cfg.target, cfg.anomaly_threshold,
                                                         cfg.anomaly_max_iter, log=out.log)
    out.log(f"Constants fitted on the normal orbits: {fmt_params(model.params_)}")

    summary = orbit_summary(sub, ANOMALY_COLUMNS)
    summary["bias"] = orbit_bias
    summary["anomalous"] = summary.index.isin(anomalous)
    pct_rows = 100 * summary.loc[summary["anomalous"], "n"].sum() / summary["n"].sum()
    out.log(f"\nAnomalous orbits: {len(anomalous)} of {len(summary)} ({100 * len(anomalous) / len(summary):.1f}%), "
            f"{pct_rows:.1f}% of rows")

    compare = summary.groupby("anomalous")[ANOMALY_COLUMNS + ["bias"]].mean().T
    compare.columns = ["normal", "anomalous"]
    compare["difference"] = compare["anomalous"] - compare["normal"]
    out.show(compare, "Orbit means: normal and anomalous orbits", digits=2)
    rho = summary[["bias", "median_corrected_echo_power"]].corr(method="spearman").iloc[0, 1]
    out.log(f"Spearman correlation (per orbit) between echo residual and noise floor: {rho:.2f}")

    sens = threshold_sensitivity(summary["bias"], summary["n"], cfg.anomaly_thresholds)
    out.show(sens, "Threshold sensitivity", digits=1)

    episodes = group_episodes(summary[summary["anomalous"]])
    out.show(episodes, f"The anomalous orbits form {len(episodes)} episodes", digits=2)

    out.save_table(summary.loc[summary["anomalous"], ["year", "n", "bias", "F10_7_index"]].reset_index(),
                   "anomalous_orbits.csv")
    out.save_table(summary.reset_index(), "orbit_summary.csv")
    out.save_table(sens, "anomaly_thresholds.csv")
    out.save_table(episodes, "anomaly_episodes.csv")
    out.update_summary({"anomalies": {
        "formula": formula.name, "threshold_dB": cfg.anomaly_threshold,
        "anomalous_orbits": len(anomalous), "total_orbits": len(summary), "pct_rows": pct_rows,
        "episodes": len(episodes),
        "echo_difference_dB": compare.loc[TARGET, "difference"],
        "noise_difference_dB": compare.loc["median_corrected_echo_power", "difference"],
        "spearman_residual_noise": rho,
        "formula_constants": dict(model.params_),
    }})


# 06 attenuation

def step_attenuation(cfg, out):
    """attenuation = echo predicted by the night law - measured echo [dB], positive = signal absorbed.
    The night law is fitted on reference_experiment and applied to every row; the result is the target
    of the search and validate steps of this experiment."""
    if not (cfg.reference_formula and cfg.reference_experiment):
        raise SystemExit(f"{cfg.name}: reference_formula and reference_experiment missing in config.py")
    df = data.load_with_features()
    night = data.subset(df, get_experiment(cfg.reference_experiment))
    ref = FormulaModel(load_formula(cfg.reference_formula), fit_sample=300_000).fit(night, night[TARGET])
    out.log(f"Reference night law: {cfg.reference_formula}, fitted on {cfg.reference_experiment} "
            f"({len(night):,} rows)")
    out.log(f"  {fmt_params(ref.params_)}")

    df["expected_no_iono"] = ref.predict(df)
    df["attenuation"] = df["expected_no_iono"] - df[TARGET]
    path = data.derived_target_file(cfg.name, "attenuation")
    path.parent.mkdir(parents=True, exist_ok=True)
    df[["expected_no_iono", "attenuation"]].to_parquet(path)
    out.log(f"saved: results/{cfg.name}/{path.name}")

    anomalous = data.load_anomalous_orbits(cfg.exclude_anomalies_of) if cfg.exclude_anomalies_of else []
    clean = df[~df[ORBIT].isin(anomalous)].copy()

    chk = clean.loc[clean.index.intersection(night.index)].groupby("freq_mhz")["attenuation"].agg(
        rows="size", mean="mean", median="median")
    out.show(chk, "Check: attenuation at night (should be ~0)", digits=2)

    clean["sun_bin"] = (clean["Sun_elevation_angle"] // 5) * 5 + 2.5
    table = clean.pivot_table(index="sun_bin", columns="freq_mhz", values="attenuation", aggfunc="median")
    counts = clean.pivot_table(index="sun_bin", columns="freq_mhz", values="attenuation", aggfunc="size")
    table = table.where(counts >= 500)
    out.show(table, "Median attenuation [dB] by Sun elevation bin (rows) and frequency (empty if < 500 rows)",
             digits=1)
    out.save_table(table.reset_index(), "attenuation_by_sun.csv")

    day = clean[clean["Sun_elevation_angle"] > 5]
    by_sun = day.pivot_table(index="sun_bin", columns="freq_mhz", values="attenuation", aggfunc="median")
    ratio = None
    if 4.0 in by_sun and 5.0 in by_sun:
        ratio = (by_sun[4.0] / by_sun[5.0]).dropna()
        out.show(ratio.to_frame("att4/att5"), "Ratio att(4 MHz)/att(5 MHz) by Sun bin "
                                              "(1/f^2 predicts 1.56, 1/f predicts 1.25)", digits=2)

    f107 = day.pivot_table(index=pd.cut(day["f107_at_mars"], [20, 30, 35, 40, 50, 100]), columns="freq_mhz",
                           values="attenuation", aggfunc="median", observed=True)
    out.show(f107, "Median attenuation with Sun > 5° by F10.7 at Mars and frequency", digits=1)

    lit = clean[clean["cos_sza"] > cfg.cos_sza_min]
    out.log(f"\nLit rows (cos χ > {cfg.cos_sza_min}) without anomalous orbits: {data.count_by(lit)}")
    out.update_summary({"attenuation": {
        "reference_formula": cfg.reference_formula,
        "reference_params": dict(ref.params_),
        "night_mean_dB": chk["mean"].to_dict(),
        "median_ratio_4_5": float(ratio.median()) if ratio is not None and len(ratio) else None,
        "lit_rows": data.count_by(lit),
    }})
    out.save_table(data.binned_median(clean["cos_sza"], clean["attenuation"], np.linspace(-0.75, 0.6, 28),
                                      clean["freq_mhz"]).rename(columns={"group": "freq_mhz", "x": "cos_sza"}),
                   "attenuation_by_cos_sza.csv")


# 07 exponents

def step_exponents(cfg, out):
    """95% confidence intervals of the attenuation law by bootstrap over orbits (rows of one orbit are
    not independent, so orbits are resampled). Sensitivity check: fit without 1.8 MHz."""
    if not cfg.exponent_formula:
        raise SystemExit(f"{cfg.name}: exponent_formula missing in config.py")
    sub = data.subset(data.load_with_features(), cfg)
    formula = load_formula(cfg.exponent_formula)
    names = list(formula.params)
    out.log(f"Formula: {formula.name}  ({formula.expression})")
    out.log(f"Rows: {len(sub):,}, orbits: {sub[ORBIT].nunique()}, per frequency: {data.count_by(sub)}")

    full = FormulaModel(formula, fit_sample=300_000).fit(sub, sub[cfg.target])
    no18 = sub[sub["freq_mhz"] != 1.8]
    without_18 = FormulaModel(formula, fit_sample=300_000).fit(no18, no18[cfg.target])

    rng = np.random.default_rng(0)
    orbits = sub[ORBIT].unique()
    rows_of = sub.groupby(ORBIT).indices
    boot = []
    t0 = time.time()
    for b in range(cfg.bootstrap):
        chosen = rng.choice(orbits, size=len(orbits), replace=True)
        sample = sub.iloc[np.concatenate([rows_of[o] for o in chosen])]
        boot.append(FormulaModel(formula, fit_sample=cfg.bootstrap_sample, seed=b).fit(sample, sample[cfg.target]).params_)
    boot = pd.DataFrame(boot)[names]
    out.log(f"Bootstrap: {cfg.bootstrap} orbit resamplings in {(time.time() - t0) / 60:.1f} minutes")

    table = pd.DataFrame({"estimate": pd.Series(full.params_), "ci95_low": boot.quantile(0.025),
                          "ci95_high": boot.quantile(0.975), "without_1.8MHz": pd.Series(without_18.params_)}).loc[names]
    fold_path = RESULTS_DIR / cfg.name / "validate_params.csv"
    if fold_path.exists():
        fp = pd.read_csv(fold_path)
        fp = fp[fp["model"] == formula.name].groupby("parameter")["value"]
        table["fold_min"], table["fold_max"] = fp.min(), fp.max()
    table["theory"] = pd.Series(cfg.theory)
    table["theory_excluded"] = [("yes" if (t < lo or t > hi) else "no") if pd.notna(t) else ""
                                for t, lo, hi in zip(table["theory"], table["ci95_low"], table["ci95_high"])]
    out.show(table, "Attenuation law parameters (95% CI from the orbit bootstrap)", digits=3)
    out.save_table(table.reset_index(names="parameter"), "exponents.csv")
    out.save_table(boot, "exponents_bootstrap.csv")

    if "s" in full.params_:
        s = full.params_["s"]
        out.log(f"Attenuation starts at cos χ = {-s:.3f}, i.e. Sun elevation {np.degrees(np.arcsin(-s)):.1f}° "
                f"at the surface")
    out.update_summary({"exponents": {
        "formula": formula.name,
        "params": {p: {"estimate": r["estimate"], "ci95": [r["ci95_low"], r["ci95_high"]], "theory": r["theory"]}
                   for p, r in table.iterrows()},
    }})


STEPS = {"explore": step_explore, "search": step_search, "validate": step_validate, "residuals": step_residuals,
         "anomalies": step_anomalies, "attenuation": step_attenuation, "exponents": step_exponents}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", required=True, choices=list(EXPERIMENTS))
    parser.add_argument("--step", required=True, nargs="+", choices=list(STEPS))
    args = parser.parse_args()

    cfg = get_experiment(args.exp)
    for step in args.step:
        out = ResultsWriter(cfg.name, cfg, step=step)
        out.log(cfg.description)
        STEPS[step](cfg, out)


if __name__ == "__main__":
    main()
