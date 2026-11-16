from __future__ import annotations

import pandas as pd

import data
from config import DATA_CACHE, ORBIT, TARGET
from results_writer import ResultsWriter

# physically plausible ranges: a value outside means a bug in the feature computation
EXPECTED_RANGES = {
    "sin_ls": (-1, 1),
    "cos_ls": (-1, 1),
    "altitude_km": (200, 1000),
    "log10_altitude_km": (2.3, 3.0),
    "freq_mhz": (1.8, 5.0),
    "wavelength_m": (59, 167),
    "roughness_over_lambda": (0, 5),
    "cos_sza": (-1, 1),
    "msd_au": (1.38, 1.67),
    "f107_at_mars": (15, 150),
    "dipole_dev": (0, 90),
}


def main():
    out = ResultsWriter("dataset", step="prepare_data")

    created = not DATA_CACHE.exists()
    raw = data.load()
    out.log(f"Parquet cache {'created' if created else 'already present'}: {DATA_CACHE}")
    out.log(f"Raw dataset: {raw.shape[0]:,} rows, {raw.shape[1]} columns")

    columns = pd.DataFrame({
        "dtype": raw.dtypes.astype(str),
        "missing": raw.isna().sum(),
        "pct_missing": 100 * raw.isna().mean(),
        "distinct": raw.nunique(),
        "min": raw.min(),
        "max": raw.max(),
    })
    out.show(columns, "Raw columns", digits=2)
    out.save_table(columns, "columns.csv", index=True)

    freq = (raw["frequency"] / 1e6).round(1)
    out.show(pd.crosstab(raw["band"], freq).rename_axis(index="band", columns="frequency MHz"),
             "Rows per band and frequency", digits=0)

    per_instant = raw.groupby("ephemeris_time").size().value_counts().sort_index()
    df = data.add_features(raw)
    summary = {
        "rows": len(raw),
        "columns": raw.shape[1],
        "instants": raw["ephemeris_time"].nunique(),
        "instants_with_two_rows": per_instant.get(2, 0),
        "years": [df["year"].min(), df["year"].max()],
        "orbits": raw[ORBIT].nunique(),
        "latitude": [raw["latitude"].min(), raw["latitude"].max()],
        "rows_per_frequency": data.count_by(df),
        "mean_target_dB": raw[TARGET].mean(),
    }
    out.log(f"\nDistinct instants: {summary['instants']:,} "
            f"({summary['instants_with_two_rows']:,} with two rows: both bands transmit together)")
    out.log(f"Period: {df['year'].min():.2f} - {df['year'].max():.2f}")
    out.log(f"Orbits: {summary['orbits']} ({int(raw[ORBIT].min())} to {int(raw[ORBIT].max())})")
    out.log(f"Latitude: {raw['latitude'].min():.1f}° to {raw['latitude'].max():.1f}° (south pole only)")
    out.log(f"Target ({TARGET}): mean {raw[TARGET].mean():.2f} dB, "
            f"range {raw[TARGET].min():.1f} to {raw[TARGET].max():.1f} dB")

    new = [c for c in df.columns if c not in raw.columns]
    out.log(f"\nPhysical features added: {len(new)}\n{data.describe_features(new)}")

    stats = df[list(EXPECTED_RANGES)].describe().T[["mean", "std", "min", "max"]]
    stats["expected_min"] = [EXPECTED_RANGES[c][0] for c in stats.index]
    stats["expected_max"] = [EXPECTED_RANGES[c][1] for c in stats.index]
    stats["ok"] = (stats["min"] >= stats["expected_min"]) & (stats["max"] <= stats["expected_max"])
    out.show(stats, "Feature values against the expected ranges")
    out.save_table(stats, "feature_check.csv", index=True)

    bad = list(stats.index[~stats["ok"]])
    out.log("\nAll features are within the expected ranges." if not bad else f"\nWARNING, out of range: {bad}")
    out.log(f"Columns never used as inputs (derived from the target): {data.LEAKAGE}")
    summary["features_ok"] = not bad
    out.update_summary(summary)


if __name__ == "__main__":
    main()
