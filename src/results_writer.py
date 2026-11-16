from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from config import RESULTS_DIR


def _to_json(value):
    if isinstance(value, dict):
        return {str(k): _to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    return value


class ResultsWriter:
    def __init__(self, name, cfg=None, step=""):
        self.dir = RESULTS_DIR / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.dir / "log.txt", "a", encoding="utf-8")
        self.log(f"\n=== {datetime.now():%Y-%m-%d %H:%M}  {name}  {step} ===")
        if is_dataclass(cfg):
            with open(self.dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(_to_json(asdict(cfg)), f, indent=2, ensure_ascii=False)

    def log(self, message=""):
        print(message)
        self._log_file.write(str(message) + "\n")
        self._log_file.flush()

    def path(self, filename):
        return self.dir / filename

    def save_table(self, df, filename, index=False):
        path = self.path(filename)
        if path.suffix == ".parquet":
            df.to_parquet(path, index=index)
        else:
            df.to_csv(path, index=index)
        self.log(f"saved: results/{self.dir.name}/{filename}")
        return path

    def show(self, df, title=None, digits=3):
        if title:
            self.log(f"\n{title}")
        with pd.option_context("display.width", 250, "display.max_columns", 50, "display.max_colwidth", 140):
            self.log(df.round(digits).to_string())

    def update_summary(self, values):
        path = self.path("summary.json")
        summary = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        summary.update(_to_json(values))
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
