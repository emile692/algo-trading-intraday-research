"""Rebuild the pre-fix ORB journal with the frozen backtester source.

The legacy engine is kept solely to audit the historical result. It must not
be used for new strategy research.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import src.analytics.orb_research.exits as exits  # noqa: E402
from src.analytics.orb_research.campaign import (  # noqa: E402
    _build_incremental_experiments,
    _evaluate_experiment,
    _split_sessions,
)
from src.analytics.orb_research.features import (  # noqa: E402
    attach_daily_reference,
    build_candidate_universe,
    build_daily_reference,
    prepare_minute_dataset,
)
from src.analytics.orb_research.types import (  # noqa: E402
    BaselineEnsembleConfig,
    BaselineEntryConfig,
    CampaignContext,
)


DATASET = REPO / "data/processed/parquet/MNQ_c_0_1m_20260321_094501.parquet"
KEYS = ("n_trades", "net_pnl", "sharpe_ratio", "max_drawdown", "profit_factor", "win_rate", "expectancy")


def main() -> None:
    spec = importlib.util.spec_from_file_location("frozen_legacy_orb_backtester", HERE / "legacy_backtester.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load frozen legacy backtester")
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    exits.run_backtest = legacy.run_backtest

    entry, ensemble = BaselineEntryConfig(), BaselineEnsembleConfig()
    experiment = _build_incremental_experiments(entry, ensemble)[0]
    minute = prepare_minute_dataset(DATASET, entry, (10, 14, 20, 30))
    minute = attach_daily_reference(minute, build_daily_reference(minute))
    candidates = build_candidate_universe(minute, baseline_entry=entry)
    sessions = sorted(pd.to_datetime(minute["session_date"]).dt.date.unique())
    is_sessions, oos_sessions = _split_sessions(sessions, 0.70)
    context = CampaignContext(
        all_sessions=sessions,
        is_sessions=is_sessions,
        oos_sessions=oos_sessions,
        minute_df=minute,
        candidate_base_df=candidates,
        daily_patterns=None,
    )
    row, details = _evaluate_experiment(experiment, context, bootstrap_paths=300, random_seed=42, keep_details=True)
    if row["status"] != "ok" or details is None:
        raise RuntimeError(f"Legacy ORB replay failed: {row['status']}")

    audit = json.loads((HERE / "audit.json").read_text(encoding="utf-8"))
    for scope in ("overall", "is", "oos"):
        for key in KEYS:
            expected = float(audit["archived_metrics"][scope][key])
            observed = float(row[f"{scope}_{key}"])
            if abs(expected - observed) > 1e-6:
                raise ValueError(f"Legacy replay mismatch: {scope}.{key}: {observed} vs {expected}")

    details["trades"].to_csv(HERE / "archived_trades_before_fix.csv", index=False)
    payload = {
        "source": "frozen legacy_backtester.py + unchanged baseline signal generation",
        "split": {"is_end": str(is_sessions[-1]), "oos_start": str(oos_sessions[0])},
        "metrics": audit["archived_metrics"],
        "trade_count": len(details["trades"]),
    }
    (HERE / "archived_summary_before_fix.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("legacy trade log reproduced", len(details["trades"]))


if __name__ == "__main__":
    main()
