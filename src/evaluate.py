from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from formulas import FormulaModel

BASELINES = {
    "constant": lambda: DummyRegressor(strategy="median"),  # the median minimises the MAE
    "linear": LinearRegression,
    "GBM": lambda: HistGradientBoostingRegressor(max_iter=300, random_state=0),
}


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def mae_by_group(y_true, y_pred, groups):
    err = pd.Series(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)))
    return err.groupby(np.asarray(groups)).mean().to_dict()


def model_factory(name, features):
    """(function that builds a fresh model, input columns) for a baseline or a formula."""
    if name in BASELINES:
        return BASELINES[name], features
    model = FormulaModel(name)
    return (lambda: FormulaModel(name)), model.formula.features


def _train_rows(df, train_idx, train_sample, seed):
    train = df.loc[train_idx]
    if train_sample is not None and len(train) > train_sample:
        train = train.sample(train_sample, random_state=seed)
    return train


def cross_validate(make_model, df, features, target, folds, train_sample=None, group_by=None, seed=0, log=print):
    """Fit a fresh model per fold and score it on the full test fold.

    Returns (one row per fold, fitted parameters per fold or None).
    """
    rows, params = [], []
    for k, (train_idx, test_idx) in enumerate(folds):
        train = _train_rows(df, train_idx, train_sample, seed)
        test = df.loc[test_idx]

        t0 = time.time()
        model = make_model()
        model.fit(train[features], train[target])
        pred = model.predict(test[features])

        row = {"fold": k, "n_test": len(test), "mae": mae(test[target], pred), "seconds": time.time() - t0}
        if group_by is not None:
            for g, v in mae_by_group(test[target], pred, test[group_by]).items():
                row[f"mae_{group_by}={g:g}"] = v
        rows.append(row)
        if hasattr(model, "params_"):
            params.append({"fold": k, **model.params_})
        log(f"  fold {k}: MAE {row['mae']:.3f}  ({row['seconds']:.0f} s)")
    return pd.DataFrame(rows), (pd.DataFrame(params).set_index("fold") if params else None)


def out_of_fold_predictions(make_model, df, features, target, folds, train_sample=None, seed=0):
    pred = pd.Series(np.nan, index=df.index)
    for train_idx, test_idx in folds:
        train = _train_rows(df, train_idx, train_sample, seed)
        model = make_model()
        model.fit(train[features], train[target])
        pred.loc[test_idx] = model.predict(df.loc[test_idx, features])
    return pred
