"""Replay the frozen nine-sleeve book through 2026-10-02.

No parameter is selected or changed here.  The locked daily ledger remains
authoritative through 2026-03-19; newly downloaded MNQ/MGC data are used only
after that cutoff and joined to the already frozen MBT vote-ensemble ledger.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FIGURES = OUT / "figures"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".matplotlib"))

from research.baseline_backtests.baseline_engine import (  # noqa: E402
    load_frozen_specs,
    run_baseline_backtest,
)
from research.locked_four_sleeve_portfolio import run_locked_four_sleeve_portfolio as mnq_lock  # noqa: E402
from research.mgc_baseline_backtests.run_mgc_baselines import prepare_mgc_data  # noqa: E402
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


MNQ_DATA = ROOT / "data/processed/parquet/MNQ_c_0_1m_20190505_20261003.parquet"
MGC_DATA = ROOT / "data/processed/parquet/MGC_v_0_1m_20150101_20261003.parquet"
LOCKED_DAILY = ROOT / "research/locked_nine_sleeve_portfolio/daily_pnl.csv"
MBT_DAILY = ROOT / "research/mbt_three_sleeve_freeze_20261004/vote_ensemble/daily_ensemble_results.csv"
MGC_SPECS = ROOT / "research/mgc_final_two_sleeve_lock_20260926/frozen_candidate_specs"
DOWNLOAD_META = ROOT / "data/downloaded/nine_sleeve_paper_extension_20261003.metadata.json"

RESEARCH_END = pd.Timestamp("2026-03-19")
FORWARD_REPLAY_START = pd.Timestamp("2026-03-20")
NINE_SLEEVE_FREEZE_DATE = pd.Timestamp("2026-10-04")
PAPER_START = pd.Timestamp("2026-10-05")
DATA_END = pd.Timestamp("2026-10-02")

SLEEVES = [
    "mnq_orb_vwap_atr",
    "mnq_volatility_breakout",
    "mnq_compression_breakout",
    "mnq_atr_trend_flip",
    "mgc_volatility_breakout",
    "mgc_gap_continuation",
    "mbt_rolling_range_breakout",
    "mbt_trend_pullback_ma_structure",
    "mbt_volatility_breakout",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_orb() -> tuple[pd.Series, pd.DataFrame]:
    entry = BaselineEntryConfig()
    ensemble = BaselineEnsembleConfig()
    experiment = _build_incremental_experiments(entry, ensemble)[0]
    if experiment.name != "baseline_fixed":
        raise RuntimeError("Unexpected retained ORB experiment")
    minute = prepare_minute_dataset(MNQ_DATA, entry, (10, 14, 20, 30))
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
    row, details = _evaluate_experiment(
        experiment, context, bootstrap_paths=1, random_seed=20261013, keep_details=True
    )
    if row["status"] != "ok" or details is None:
        raise RuntimeError(f"ORB replay failed: {row['status']}")
    trades = details["trades"].copy()
    trades["session_date"] = pd.to_datetime(trades["session_date"])
    pnl = trades.groupby("session_date")["net_pnl_usd"].sum()
    return pnl, trades


def run_mnq_alpha() -> tuple[dict[str, pd.Series], pd.DatetimeIndex, dict[str, pd.DataFrame]]:
    config = json.loads(mnq_lock.CONFIG_PATH.read_text(encoding="utf-8"))
    config["end_date"] = DATA_END.date().isoformat()
    mnq_lock.DATA_PATH = MNQ_DATA
    runs, trades, _ = mnq_lock.run_alpha_sleeves(config)
    pnl = {}
    for name in ("volatility_breakout", "compression_breakout", "atr_trend_flip"):
        daily = runs[name].daily.copy()
        daily["session_date"] = pd.to_datetime(daily["session_date"])
        pnl[f"mnq_{name}"] = daily.set_index("session_date")["daily_pnl_usd"].astype(float)
    calendar = pd.DatetimeIndex(pd.to_datetime(runs["volatility_breakout"].daily["session_date"]))
    return pnl, calendar, trades


def run_mgc() -> tuple[dict[str, pd.Series], pd.DatetimeIndex, dict[str, pd.DataFrame]]:
    start = pd.Timestamp("2019-05-06").date()
    end = DATA_END.date()
    prepared, _ = prepare_mgc_data(MGC_DATA, start, end)
    result = {}
    trades = {}
    calendar = None
    for mechanism in ("volatility_breakout", "gap_continuation"):
        spec = json.loads((MGC_SPECS / f"{mechanism}.json").read_text(encoding="utf-8"))
        run = run_baseline_backtest(spec, prepared, evaluation_start=start, evaluation_end=end)
        daily = run.daily.copy()
        daily["session_date"] = pd.to_datetime(daily["session_date"])
        result[f"mgc_{mechanism}"] = daily.set_index("session_date")["daily_pnl_usd"].astype(float)
        trades[f"mgc_{mechanism}"] = run.trades.copy()
        if calendar is None:
            calendar = pd.DatetimeIndex(daily["session_date"])
    assert calendar is not None
    return result, calendar, trades


def metrics(pnl: pd.Series) -> dict[str, float | int | str]:
    pnl = pnl.astype(float)
    equity = pnl.cumsum()
    drawdown = equity - equity.cummax().clip(lower=0.0)
    std = float(pnl.std(ddof=1))
    sharpe = float(pnl.mean() / std * math.sqrt(252)) if std > 0 else np.nan
    trough = drawdown.idxmin()
    peak = equity.loc[:trough].idxmax()
    return {
        "start": str(pnl.index.min().date()),
        "end": str(pnl.index.max().date()),
        "sessions": int(len(pnl)),
        "net_pnl_usd": float(pnl.sum()),
        "sharpe_zero_days": sharpe,
        "max_drawdown_usd": float(drawdown.min()),
        "max_drawdown_peak": str(peak.date()),
        "max_drawdown_trough": str(trough.date()),
        "worst_day_usd": float(pnl.min()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    orb_pnl, orb_trades = run_orb()
    mnq_pnl, mnq_calendar, mnq_trades = run_mnq_alpha()
    mgc_pnl, mgc_calendar, mgc_trades = run_mgc()

    locked = pd.read_csv(LOCKED_DAILY, parse_dates=["session_date"]).set_index("session_date")
    mbt_raw = pd.read_csv(MBT_DAILY, parse_dates=["session_date"])
    mbt_names = {
        "rolling_range_breakout": "mbt_rolling_range_breakout",
        "trend_pullback_ma_structure": "mbt_trend_pullback_ma_structure",
        "volatility_breakout": "mbt_volatility_breakout",
    }
    mbt = (
        mbt_raw[
            mbt_raw["cost_scenario"].eq("nominal")
            & mbt_raw["sleeve"].isin(mbt_names)
        ]
        .pivot_table(index="session_date", columns="sleeve", values="daily_pnl_usd", aggfunc="sum")
        .rename(columns=mbt_names)
        .fillna(0.0)
    )

    extension_calendar = mnq_calendar.union(mgc_calendar).union(mbt.index)
    extension_calendar = extension_calendar[
        (extension_calendar >= FORWARD_REPLAY_START) & (extension_calendar <= DATA_END)
    ]
    extension = pd.DataFrame(0.0, index=extension_calendar, columns=SLEEVES)
    extension.index.name = "session_date"
    all_new = {"mnq_orb_vwap_atr": orb_pnl, **mnq_pnl, **mgc_pnl}
    for sleeve, series in all_new.items():
        common = extension.index.intersection(series.index)
        extension.loc[common, sleeve] = series.loc[common].astype(float)
    for sleeve in mbt.columns:
        common = extension.index.intersection(mbt.index)
        extension.loc[common, sleeve] = mbt.loc[common, sleeve].astype(float)
    extension["portfolio_pnl_usd"] = extension[SLEEVES].sum(axis=1)

    full = pd.concat([locked.loc[:RESEARCH_END], extension]).sort_index()
    if full.index.duplicated().any():
        raise RuntimeError("Duplicate session dates in extended ledger")
    full["portfolio_pnl_usd"] = full[SLEEVES].sum(axis=1)
    full.to_csv(OUT / "daily_nine_sleeves_through_20261002.csv", float_format="%.8f")

    summary = {
        "historical_through_research_cutoff": metrics(full.loc[:RESEARCH_END, "portfolio_pnl_usd"]),
        "forward_replay_20260320_20261002": metrics(full.loc[FORWARD_REPLAY_START:, "portfolio_pnl_usd"]),
        "full_20240102_20261002": metrics(full["portfolio_pnl_usd"]),
    }
    sleeve_forward = {s: metrics(full.loc[FORWARD_REPLAY_START:, s]) for s in SLEEVES}

    # The frozen common ledger must remain byte-for-byte equivalent in values.
    locked_reproduced = bool(
        np.allclose(
            full.loc[locked.index, SLEEVES + ["portfolio_pnl_usd"]].to_numpy(float),
            locked[SLEEVES + ["portfolio_pnl_usd"]].to_numpy(float),
            rtol=0,
            atol=1e-8,
        )
    )
    if not locked_reproduced:
        raise RuntimeError("Locked common ledger changed")

    cumulative = full["portfolio_pnl_usd"].cumsum()
    drawdown = cumulative - cumulative.cummax().clip(lower=0.0)
    fig, (ax, dd_ax) = plt.subplots(
        2, 1, figsize=(15, 8), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.08},
    )
    ax.axvspan(full.index.min(), RESEARCH_END, color="#dbeafe", alpha=0.65)
    ax.axvspan(FORWARD_REPLAY_START, DATA_END, color="#fef3c7", alpha=0.70)
    ax.plot(full.index, cumulative, color="#102f55", lw=2.0)
    ax.axvline(FORWARD_REPLAY_START, color="#b45309", ls="--", lw=1.2)
    ax.axhline(0, color="#6b7280", lw=0.8)
    ax.set_title("Book 9 sleeves — historique complet jusqu'au 2 octobre 2026")
    ax.set_ylabel("PnL net cumule (USD)")
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax.grid(alpha=0.20)
    ax.text(pd.Timestamp("2025-01-15"), cumulative.max() * 0.96, "Historique de recherche\nfin: 19 mars 2026", ha="center", va="top")
    ax.text(pd.Timestamp("2026-06-15"), cumulative.max() * 0.96, "Extension figee sans retuning\n20 mars – 2 octobre 2026", ha="center", va="top")

    dd_ax.fill_between(full.index, drawdown.to_numpy(), 0, color="#b91c1c", alpha=0.28)
    dd_ax.plot(full.index, drawdown, color="#991b1b", lw=0.9)
    dd_ax.set_ylabel("Drawdown")
    dd_ax.set_xlabel("Le paper trading officiel du book 9 sleeves commence le 5 octobre 2026")
    dd_ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    dd_ax.grid(alpha=0.20)
    fig.savefig(FIGURES / "nine_sleeve_equity_through_20261002.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "campaign": "nine_sleeve_paper_extension_20261013",
        "rules_changed": False,
        "research_data_end": str(RESEARCH_END.date()),
        "forward_replay_start": str(FORWARD_REPLAY_START.date()),
        "nine_sleeve_freeze_date": str(NINE_SLEEVE_FREEZE_DATE.date()),
        "official_paper_start": str(PAPER_START.date()),
        "data_end": str(DATA_END.date()),
        "governance_note": (
            "2026-03-20..2026-10-02 is a frozen forward replay for the six MNQ/MGC sleeves, "
            "but not pristine paper data for the three MBT sleeves, whose freeze used data through 2026-10-02. "
            "The official nine-sleeve paper phase begins on 2026-10-05."
        ),
        "inputs": {
            "mnq": {"path": str(MNQ_DATA.relative_to(ROOT)), "sha256": sha256(MNQ_DATA)},
            "mgc": {"path": str(MGC_DATA.relative_to(ROOT)), "sha256": sha256(MGC_DATA)},
            "mbt_daily": {"path": str(MBT_DAILY.relative_to(ROOT)), "sha256": sha256(MBT_DAILY)},
            "download_metadata": {"path": str(DOWNLOAD_META.relative_to(ROOT)), "sha256": sha256(DOWNLOAD_META)},
        },
        "validation": {
            "locked_common_ledger_reproduced": locked_reproduced,
            "orb_total_trades": int(len(orb_trades)),
            "mnq_alpha_trades": {k: int(len(v)) for k, v in mnq_trades.items()},
            "mgc_trades": {k: int(len(v)) for k, v in mgc_trades.items()},
        },
        "summary": summary,
        "forward_by_sleeve": sleeve_forward,
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    def row(label: str, values: dict) -> str:
        return (
            f"| {label} | {values['start']} | {values['end']} | {values['sessions']} | "
            f"{values['net_pnl_usd']:,.2f} | {values['sharpe_zero_days']:.3f} | "
            f"{values['max_drawdown_usd']:,.2f} | {values['worst_day_usd']:,.2f} |"
        )

    report = [
        "# Book neuf sleeves — extension jusqu'au 2 octobre 2026", "",
        "## Gouvernance des periodes", "",
        "- Fin du ledger historique de recherche commun : **19 mars 2026**.",
        "- Extension MNQ/MGC sans retuning : **20 mars au 2 octobre 2026**.",
        "- Cette extension est un forward replay fige pour les six sleeves MNQ/MGC, mais pas un paper OOS vierge pour MBT : les trois ensembles MBT ont ete figes avec des donnees allant jusqu'au 2 octobre.",
        "- Freeze officiel du book neuf sleeves : **4 octobre 2026**.",
        "- Debut officiel du paper trading prospectif : **5 octobre 2026**, premiere seance apres le freeze. Aucun chiffre jusqu'au 2 octobre ne doit etre etiquete comme paper neuf-sleeves.", "",
        "## Performance", "",
        "| Periode | Debut | Fin | Seances | PnL net USD | Sharpe | Max DD USD | Pire jour USD |",
        "|---|---|---|---:|---:|---:|---:|---:|",
        row("Historique verrouille", summary["historical_through_research_cutoff"]),
        row("Extension gelee", summary["forward_replay_20260320_20261002"]),
        row("Total", summary["full_20240102_20261002"]), "",
        "![Equity du book](figures/nine_sleeve_equity_through_20261002.png)", "",
        "## Acquisition", "",
        "Deux appels payants Databento ont complete MNQ.c.0 et MGC.v.0. Cout pre-requete total estime : **0,739163 USD**. La cle API n'est stockee dans aucun fichier.",
        "Les avertissements de qualite fournisseur sont conserves dans l'audit d'acquisition; les ledgers historiques verrouilles n'ont pas ete reecrits.", "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
