"""Compact research campaign for volume climax pullback contrarian family."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.analytics.metrics import compute_metrics
from src.data.cleaning import clean_ohlcv
from src.data.loader import load_ohlcv_file
from src.data.resampling import resample_ohlcv
from src.engine.volume_climax_pullback_backtester import run_volume_climax_pullback_backtest
from src.engine.vwap_backtester import build_execution_model_for_profile
from src.strategy.volume_climax_pullback import (
    apply_session_overlay,
    build_compact_variants,
    build_signal_frame,
    prepare_volume_climax_features,
)

SYMBOLS = ("MNQ", "MES", "M2K", "MGC")
TIMEFRAMES = {"5m": "5min", "1h": "1h"}


def _latest_path_for_symbol(symbol: str) -> Path:
    files = sorted(Path("data/processed/parquet").glob(f"{symbol}_c_0_1m_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No input dataset found for {symbol} in data/processed/parquet.")
    return files[-1]


def _split_sessions(signal_df: pd.DataFrame, ratio: float = 0.7) -> tuple[list, list]:
    sessions = sorted(pd.to_datetime(signal_df["session_date"]).dt.date.unique())
    cut = max(1, int(len(sessions) * ratio))
    cut = min(cut, len(sessions) - 1)
    return sessions[:cut], sessions[cut:]


def _summarize(trades: pd.DataFrame, signal_df: pd.DataFrame, sessions: list) -> dict:
    m = compute_metrics(trades, signal_df=signal_df, session_dates=sessions)
    return {
        "net_pnl": float(m.get("cumulative_pnl", 0.0)),
        "profit_factor": float(m.get("profit_factor", 0.0)),
        "sharpe": float(m.get("sharpe_ratio", 0.0)),
        "max_drawdown": float(m.get("max_drawdown", 0.0)),
        "nb_trades": int(m.get("n_trades", 0)),
        "avg_trade": float(trades["net_pnl_usd"].mean()) if not trades.empty else 0.0,
        "expectancy": float(m.get("expectancy", 0.0)),
        "hit_rate": float(m.get("win_rate", 0.0)),
    }


def run_campaign(output_dir: Path) -> Path:
    run_dir = output_dir / f"volume_climax_pullback_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    variants = build_compact_variants(TIMEFRAMES.keys())
    rows: list[dict] = []

    for symbol in SYMBOLS:
        src = _latest_path_for_symbol(symbol)
        raw = clean_ohlcv(load_ohlcv_file(src))
        exec_model, instrument = build_execution_model_for_profile(symbol=symbol, profile_name="repo_realistic")

        for timeframe, rule in TIMEFRAMES.items():
            bars = resample_ohlcv(raw, rule=rule)
            for variant in [v for v in variants if v.timeframe == timeframe]:
                scoped = apply_session_overlay(bars, variant.session_overlay)
                if len(scoped) < max(150, variant.volume_lookback + 10):
                    continue
                features = prepare_volume_climax_features(scoped)
                signal_df = build_signal_frame(features, variant)
                is_sessions, oos_sessions = _split_sessions(signal_df)
                is_df = signal_df[signal_df["session_date"].isin(is_sessions)].copy()
                oos_df = signal_df[signal_df["session_date"].isin(oos_sessions)].copy()

                is_trades = run_volume_climax_pullback_backtest(is_df, variant, exec_model, instrument).trades
                oos_trades = run_volume_climax_pullback_backtest(oos_df, variant, exec_model, instrument).trades

                is_m = _summarize(is_trades, is_df, is_sessions)
                oos_m = _summarize(oos_trades, oos_df, oos_sessions)
                stability = (oos_m["sharpe"] / is_m["sharpe"]) if abs(is_m["sharpe"]) > 1e-9 else np.nan
                rows.append(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "variant_name": variant.name,
                        "family": variant.family,
                        "session_overlay": variant.session_overlay,
                        **{f"is_{k}": v for k, v in is_m.items()},
                        **{f"oos_{k}": v for k, v in oos_m.items()},
                        "stability_is_oos_sharpe_ratio": stability,
                    }
                )

    summary = pd.DataFrame(rows).sort_values(["oos_net_pnl", "oos_sharpe"], ascending=False)
    summary.to_csv(run_dir / "summary_variants.csv", index=False)

    summary.sort_values(["oos_sharpe", "oos_profit_factor"], ascending=False).to_csv(run_dir / "ranking_oos_global.csv", index=False)
    summary.sort_values(["symbol", "oos_sharpe"], ascending=[True, False]).to_csv(run_dir / "ranking_oos_by_asset.csv", index=False)
    summary.sort_values(["timeframe", "oos_sharpe"], ascending=[True, False]).to_csv(run_dir / "ranking_oos_by_timeframe.csv", index=False)

    top = summary.groupby(["family", "timeframe"], as_index=False).agg(
        oos_sharpe_mean=("oos_sharpe", "mean"),
        oos_net_pnl_mean=("oos_net_pnl", "mean"),
        count=("variant_name", "count"),
    )
    pivot = top.pivot(index="family", columns="timeframe", values="oos_sharpe_mean")
    pivot.to_csv(run_dir / "heatmap_oos_sharpe_family_timeframe.csv")

    survivor = summary[(summary["oos_profit_factor"] > 1.0) & (summary["oos_net_pnl"] > 0)].copy()
    dead = summary[(summary["oos_profit_factor"] <= 1.0) | (summary["oos_net_pnl"] <= 0)].copy()

    verdict = [
        "# Volume Climax Pullback - Verdict",
        "",
        f"- Variantes testées: **{len(summary)}**.",
        f"- Survivants crédibles (PF>1 et PnL OOS>0): **{len(survivor)}**.",
        f"- Variantes mortes: **{len(dead)}**.",
        "",
        "## Meilleure config globale",
    ]
    if not summary.empty:
        best = summary.iloc[0]
        verdict += [
            f"- {best['variant_name']} ({best['symbol']} {best['timeframe']})",
            f"- OOS pnl={best['oos_net_pnl']:.2f}, PF={best['oos_profit_factor']:.2f}, Sharpe={best['oos_sharpe']:.2f}",
        ]
    verdict += [
        "",
        "## Réponses de recherche",
        "1. Edge volume-only: voir lignes family=pure_climax dans summary_variants.csv.",
        "2. Contribution stretch: comparer climax_plus_stretch vs pure_climax.",
        "3. Impact rejection: comparer climax_plus_rejection / combined_qsr vs sans rejection.",
        "4. Actif le plus réactif: ranking_oos_by_asset.csv.",
        "5. 1h vs 5m: ranking_oos_by_timeframe.csv (comparabilité des overlays documentée dans les noms de variants).",
        "6. Pertinence portefeuille: survivants + stabilité IS/OOS.",
        "",
        "## Note comparabilité 1h",
        "- exclude_first_10m et exclude_lunch sont appliqués avec filtres horaires; sur 1h l'effet est discretisé par barres horaires et n'est pas strictement équivalent à 5m.",
    ]
    (run_dir / "final_report.md").write_text("\n".join(verdict), encoding="utf-8")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run volume climax pullback campaign.")
    parser.add_argument("--output-dir", default=None, help="Output root directory (legacy flag).")
    parser.add_argument("--output-root", default=None, help="Output root directory.")
    args = parser.parse_args()
    output_root = args.output_root or args.output_dir or "data/exports"
    out = run_campaign(Path(output_root))
    print(out)


if __name__ == "__main__":
    main()
