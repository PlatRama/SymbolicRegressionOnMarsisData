from __future__ import annotations

import warnings

import pandas as pd

from config import ORBIT
from formulas import FormulaModel


def flag_anomalous_orbits(df, formula, target, threshold=-8.0, max_iter=10, log=print):
    """Orbits whose mean residual (measured - predicted) is below `threshold` dB.

    Iterative: the formula is refitted on the normal orbits only, until the set stops changing.
    Returns (anomalous orbits, mean residual per orbit, final model).
    """
    anomalous, converged = set(), False
    for it in range(max_iter):
        normal = df[~df[ORBIT].isin(anomalous)]
        model = FormulaModel(formula).fit(normal, normal[target])
        resid = df[target] - model.predict(df)
        orbit_bias = resid.groupby(df[ORBIT]).mean()
        new = set(orbit_bias[orbit_bias < threshold].index)
        log(f"  iteration {it}: {len(new)} anomalous orbits")
        if new == anomalous:
            converged = True
            break
        anomalous = new
    if not converged:
        warnings.warn(f"no convergence in {max_iter} iterations: the set of anomalous orbits is not stable")
    return sorted(anomalous), orbit_bias, model


def orbit_summary(df, columns):
    out = df.groupby(ORBIT)[columns].mean()
    out["n"] = df.groupby(ORBIT).size()
    return out


def threshold_sensitivity(orbit_bias, orbit_sizes, thresholds=(-6, -8, -10, -12)):
    rows = []
    for t in thresholds:
        sel = orbit_bias < t
        rows.append({"threshold_dB": t, "orbits": int(sel.sum()), "pct_orbits": 100 * sel.mean(),
                     "pct_rows": 100 * orbit_sizes[sel].sum() / orbit_sizes.sum()})
    return pd.DataFrame(rows)


def group_episodes(anomalous_summary, max_gap=20):
    """Group anomalous orbits at most max_gap orbits apart (~5 days). Needs columns "year" and "bias"."""
    s = anomalous_summary.sort_index()
    episode = (s.index.to_series().diff() > max_gap).cumsum().to_numpy()
    orbits = s.index.to_series()
    return pd.DataFrame({
        "first_orbit": orbits.groupby(episode).min(),
        "last_orbit": orbits.groupby(episode).max(),
        "orbits": orbits.groupby(episode).size(),
        "year": s["year"].groupby(episode).mean(),
        "bias_dB": s["bias"].groupby(episode).mean(),
    })
