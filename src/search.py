from __future__ import annotations

import pandas as pd

from evaluate import mae, mae_by_group


def make_regressor(cfg, seed):
    from pysr import PySRRegressor  # loading Julia is slow: only when searching

    return PySRRegressor(
        niterations=cfg.pysr_iterations,
        maxsize=cfg.pysr_maxsize,
        binary_operators=cfg.pysr_binary,
        unary_operators=cfg.pysr_unary,
        nested_constraints=cfg.pysr_nested,
        constraints=cfg.pysr_constraints,
        timeout_in_seconds=cfg.pysr_timeout or None,
        elementwise_loss="L1DistLoss()",
        batching=True,
        batch_size=1000,
        temp_equation_file=True,
        progress=False,
        random_state=seed,
    )


def balanced_sample(df, n, by, seed=0):
    """About n rows with the same count for every value of `by`, so rare groups are not ignored."""
    groups = df.groupby(by)
    per_group = n // groups.ngroups
    return pd.concat(g.sample(min(len(g), per_group), random_state=seed) for _, g in groups)


def search_sample(train, cfg, seed):
    if cfg.pysr_balance_by:
        return balanced_sample(train, cfg.pysr_sample, cfg.pysr_balance_by, seed=seed)
    return train.sample(min(cfg.pysr_sample, len(train)), random_state=seed)


def evaluate_pareto(model, features, target, datasets, group_by=None):
    """MAE of every Pareto formula on each dataset ({"train": ..., "test": ...})."""
    rows = []
    for i, eq in model.equations_.iterrows():
        row = {"complexity": eq["complexity"]}
        for name, d in datasets.items():
            pred = model.predict(d[features], index=i)
            row[f"mae_{name}"] = mae(d[target], pred)
            if group_by is not None and name == "test":
                for g, v in mae_by_group(d[target], pred, d[group_by]).items():
                    row[f"mae_test_{group_by}={g:g}"] = v
        row["equation"] = eq["equation"]
        rows.append(row)
    return pd.DataFrame(rows)


def run_search(train, test, cfg, log=print):
    """One PySR search per seed; returns all Pareto fronts in one table (column "seed")."""
    tables = []
    for seed in cfg.pysr_seeds:
        log(f"  search with seed {seed} ...")
        sample = search_sample(train, cfg, seed)
        model = make_regressor(cfg, seed)
        model.fit(sample[cfg.features], sample[cfg.target])
        pareto = evaluate_pareto(model, cfg.features, cfg.target, {"train": train, "test": test},
                                 group_by=cfg.group_by)
        pareto.insert(0, "seed", seed)
        tables.append(pareto)
    return pd.concat(tables, ignore_index=True)
