"""MNQ OR30 both-direction Ensemble ORB campaign across multiple RR values."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.orb_multi_asset_campaign import (  # noqa: E402
    BaselineSpec,
    SearchGrid,
    _data_sanity_markdown,
    _export_symbol_analysis,
    _plot_heatmap,
    analyze_symbol,
)

REPO_ROOT = ROOT
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "export" / "mnq_or30_both_rr_campaign"


@dataclass(frozen=True)
class MnqBothRrCampaignConfig:
    symbol: str = "MNQ"
    or_minutes: int = 30
    direction: str = "both"
    rr_values: tuple[float, ...] = (2.0, 3.0, 5.0, 10.0)
    is_fraction: float = 0.70
    grid: SearchGrid = SearchGrid()
    output_root: Path = DEFAULT_OUTPUT_ROOT


def _rr_tag(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "_")


def _comparison_rows(analyses: list[Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for analysis in analyses:
        rr = float(analysis.baseline.target_multiple)
        best = analysis.best_ensemble
        robust = analysis.robust_ensemble
        rows.append(
            {
                "rr": rr,
                "folder": f"rr_{_rr_tag(rr)}",
                "best_rule": best["aggregation_rule"],
                "robust_rule": robust["aggregation_rule"],
                "best_overall_score": float(best["overall_composite_score"]),
                "best_overall_sharpe": float(best["overall_sharpe"]),
                "best_overall_pf": float(best["overall_profit_factor"]),
                "best_overall_net_pnl": float(best["overall_net_pnl"]),
                "best_overall_maxdd_abs": float(best["overall_max_drawdown_abs"]),
                "best_oos_score": float(best["oos_composite_score"]),
                "best_oos_sharpe": float(best["oos_sharpe"]),
                "best_oos_pf": float(best["oos_profit_factor"]),
                "best_oos_net_pnl": float(best["oos_net_pnl"]),
                "best_oos_maxdd_abs": float(best["oos_max_drawdown_abs"]),
                "best_cell_atr": int(analysis.best_cell["atr_period"]),
                "best_cell_q_low": int(analysis.best_cell["q_low_pct"]),
                "best_cell_q_high": int(analysis.best_cell["q_high_pct"]),
                "robust_clusters": int(len(analysis.robust_clusters)),
            }
        )
    return pd.DataFrame(rows).sort_values("rr").reset_index(drop=True)


def _export_overall_heatmaps(analysis: Any, charts_dir: Path) -> None:
    symbol_lower = analysis.symbol.lower()
    _plot_heatmap(
        analysis.point_results,
        value_col="overall_sharpe",
        title=f"{analysis.symbol} Overall Sharpe",
        output_path=charts_dir / f"{symbol_lower}_heatmap_overall_sharpe.png",
        cmap="RdYlGn",
    )
    _plot_heatmap(
        analysis.point_results,
        value_col="overall_profit_factor",
        title=f"{analysis.symbol} Overall Profit Factor",
        output_path=charts_dir / f"{symbol_lower}_heatmap_overall_pf.png",
        cmap="RdYlGn",
    )
    _plot_heatmap(
        analysis.point_results,
        value_col="overall_net_pnl",
        title=f"{analysis.symbol} Overall Net PnL",
        output_path=charts_dir / f"{symbol_lower}_heatmap_overall_netpnl.png",
        cmap="RdYlGn",
    )
    _plot_heatmap(
        analysis.point_results,
        value_col="overall_max_drawdown_abs",
        title=f"{analysis.symbol} Overall |Max Drawdown|",
        output_path=charts_dir / f"{symbol_lower}_heatmap_overall_maxdd.png",
        cmap="RdYlGn",
        reverse=True,
    )
    _plot_heatmap(
        analysis.point_results,
        value_col="overall_composite_score",
        title=f"{analysis.symbol} Overall Composite Score",
        output_path=charts_dir / f"{symbol_lower}_heatmap_overall_score.png",
        cmap="RdYlGn",
    )


def _write_root_summary(compare_df: pd.DataFrame, analyses: list[Any], output_path: Path) -> None:
    best_overall = compare_df.sort_values(
        ["best_overall_score", "best_overall_pf", "best_overall_sharpe"],
        ascending=[False, False, False],
    ).iloc[0]
    best_oos = compare_df.sort_values(
        ["best_oos_score", "best_oos_pf", "best_oos_sharpe"],
        ascending=[False, False, False],
    ).iloc[0]

    lines = [
        "# MNQ OR30 Both RR Campaign",
        "",
        "- Symbol: `MNQ`",
        "- Opening range: `30 minutes`",
        "- Direction mode: `both`",
        "- Data source: processed parquet from `data/processed/parquet`",
        "- RR values tested: " + ", ".join(str(int(v) if float(v).is_integer() else v) for v in compare_df["rr"].tolist()),
        "",
        "## Comparison Table",
        "",
        compare_df.to_string(index=False),
        "",
        "## Direct Readout",
        "",
        f"- Best RR by overall score: `RR {int(best_overall['rr']) if float(best_overall['rr']).is_integer() else best_overall['rr']}` "
        f"with `{best_overall['best_rule']}` | overall score `{best_overall['best_overall_score']:.3f}` | "
        f"overall Sharpe `{best_overall['best_overall_sharpe']:.3f}` | overall PF `{best_overall['best_overall_pf']:.3f}`.",
        f"- Best RR by OOS score: `RR {int(best_oos['rr']) if float(best_oos['rr']).is_integer() else best_oos['rr']}` "
        f"with `{best_oos['best_rule']}` | OOS score `{best_oos['best_oos_score']:.3f}` | "
        f"OOS Sharpe `{best_oos['best_oos_sharpe']:.3f}` | OOS PF `{best_oos['best_oos_pf']:.3f}`.",
        "",
        "## Per-RR Folders",
        "",
    ]

    for analysis in analyses:
        rr = float(analysis.baseline.target_multiple)
        folder = f"rr_{_rr_tag(rr)}"
        lines.extend(
            [
                f"### RR {int(rr) if rr.is_integer() else rr}",
                "",
                f"- Folder: `{folder}`",
                f"- Best ensemble: `{analysis.best_ensemble['aggregation_rule']}`",
                f"- Best cell: `ATR {int(analysis.best_cell['atr_period'])} / q{int(analysis.best_cell['q_low_pct'])}/q{int(analysis.best_cell['q_high_pct'])}`",
                f"- Overall score / Sharpe / PF: `{float(analysis.best_ensemble['overall_composite_score']):.3f}` / `{float(analysis.best_ensemble['overall_sharpe']):.3f}` / `{float(analysis.best_ensemble['overall_profit_factor']):.3f}`",
                f"- OOS score / Sharpe / PF: `{float(analysis.best_ensemble['oos_composite_score']):.3f}` / `{float(analysis.best_ensemble['oos_sharpe']):.3f}` / `{float(analysis.best_ensemble['oos_profit_factor']):.3f}`",
                "",
            ]
        )

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_mnq_or30_both_rr_campaign(config: MnqBothRrCampaignConfig) -> dict[str, Path]:
    root = Path(config.output_root)
    root.mkdir(parents=True, exist_ok=True)

    analyses = []
    for rr in config.rr_values:
        rr_value = float(rr)
        rr_dir = root / f"rr_{_rr_tag(rr_value)}"
        baseline = BaselineSpec(
            or_minutes=int(config.or_minutes),
            direction=str(config.direction),
            target_multiple=rr_value,
        )
        analysis = analyze_symbol(
            symbol=config.symbol,
            baseline=baseline,
            grid=config.grid,
            is_fraction=float(config.is_fraction),
        )
        analysis.export_paths = _export_symbol_analysis(analysis, rr_dir)
        _export_overall_heatmaps(analysis, rr_dir / "charts")
        analyses.append(analysis)

    sanity_path = root / "data_sanity_check.md"
    _data_sanity_markdown(analyses, sanity_path)

    compare_df = _comparison_rows(analyses)
    compare_path = root / "rr_comparison.csv"
    compare_df.to_csv(compare_path, index=False)

    summary_path = root / "rr_summary.md"
    _write_root_summary(compare_df, analyses, summary_path)

    metadata = {
        "run_timestamp": datetime.now().isoformat(),
        "config": {
            **asdict(config),
            "output_root": str(config.output_root),
        },
        "rr_folders": {f"rr_{_rr_tag(float(a.baseline.target_multiple))}": str((root / f'rr_{_rr_tag(float(a.baseline.target_multiple))}')) for a in analyses},
    }
    metadata_path = root / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    return {
        "root": root,
        "data_sanity_check": sanity_path,
        "comparison": compare_path,
        "summary": summary_path,
        "metadata": metadata_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MNQ OR30 both-direction RR campaign.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--is-fraction", type=float, default=0.70)
    args = parser.parse_args()

    config = MnqBothRrCampaignConfig(
        output_root=Path(args.output_root),
        is_fraction=float(args.is_fraction),
    )
    artifacts = run_mnq_or30_both_rr_campaign(config)
    print(f"summary: {artifacts['summary']}")
    print(f"comparison: {artifacts['comparison']}")


if __name__ == "__main__":
    main()
