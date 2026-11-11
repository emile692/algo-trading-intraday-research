"""Paired moving-block uncertainty diagnostics for the MBT integration."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SEED = 20261004
REPS = 5000
BLOCK = 10


def sharpe(values: np.ndarray) -> float:
    std = values.std(ddof=1)
    return float(values.mean() / std * math.sqrt(252.0)) if std > 0 else float("nan")


def sample_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n - BLOCK + 1, size=math.ceil(n / BLOCK))
    return np.concatenate([np.arange(start, start + BLOCK) for start in starts])[:n]


def diagnose(frame: pd.DataFrame) -> dict[str, float | int | list[float]]:
    old = frame["current_six_equal_weight_raw"].to_numpy(float)
    new = frame["new_three_equal_weight_raw"].to_numpy(float)
    nine = frame["nine_equal_weight_raw"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    corr = np.empty(REPS)
    delta = np.empty(REPS)
    for rep in range(REPS):
        idx = sample_indices(len(frame), rng)
        corr[rep] = np.corrcoef(old[idx], new[idx])[0, 1]
        delta[rep] = sharpe(nine[idx]) - sharpe(old[idx])
    return {
        "sessions": len(frame),
        "block_length": BLOCK,
        "repetitions": REPS,
        "book_correlation": float(np.corrcoef(old, new)[0, 1]),
        "book_correlation_ci95": np.quantile(corr, [0.025, 0.975]).tolist(),
        "raw_equal_weight_sharpe_delta_6_to_9": sharpe(nine) - sharpe(old),
        "raw_equal_weight_sharpe_delta_ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
    }


def main() -> None:
    frame = pd.read_csv(HERE / "daily_aligned_nine_sleeves.csv", parse_dates=["session_date"]).set_index("session_date")
    result = {
        "common_full": diagnose(frame),
        "common_legacy_oos": diagnose(frame.loc["2025-05-29":]),
        "caveat": "Moving-block intervals quantify sampling uncertainty only; they do not repair selection bias or create independent OOS evidence.",
    }
    (HERE / "bootstrap_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
