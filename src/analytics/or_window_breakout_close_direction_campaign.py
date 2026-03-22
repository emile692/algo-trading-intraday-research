"""Simple descriptive campaign for OR-window close-breakout directionality."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.orb_multi_asset_campaign import resolve_processed_dataset
from src.config.paths import NOTEBOOKS_DIR
from src.config.settings import DEFAULT_TIMEZONE

REPO_ROOT = ROOT
DEFAULT_SYMBOLS: tuple[str, ...] = ("MES", "MNQ", "MGC", "M2K")
DEFAULT_OR_WINDOWS: tuple[int, ...] = (5, 15, 30, 60)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "export" / "or_window_close_breakout_direction_campaign"
DEFAULT_NOTEBOOK_PATH = NOTEBOOKS_DIR / "orb_or_window_breakout_close_direction_validation.ipynb"

RTH_START_MINUTE = 9 * 60 + 30
RTH_END_MINUTE = 16 * 60
CLOSE_PROXY_MINUTE = 15 * 60 + 55
BREAKOUT_DIRECTIONS = ("up", "down")


@dataclass(frozen=True)
class CampaignConfig:
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    or_windows: tuple[int, ...] = DEFAULT_OR_WINDOWS
    output_root: Path = DEFAULT_OUTPUT_ROOT
    notebook_path: Path = DEFAULT_NOTEBOOK_PATH


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0 or not math.isfinite(float(denominator)):
        return 0.0
    value = float(numerator) / float(denominator)
    return float(value) if math.isfinite(value) else 0.0


def _minute_of_day(timestamp: pd.Series) -> pd.Series:
    ts = pd.to_datetime(timestamp)
    return ts.dt.hour * 60 + ts.dt.minute


def load_processed_minute_data(path: Path | str, timezone: str = DEFAULT_TIMEZONE) -> pd.DataFrame:
    """Load a processed parquet with only the fields needed for the campaign."""
    file_path = Path(path)
    df = pd.read_parquet(file_path)
    if "timestamp" not in df.columns:
        df = df.reset_index()

    df.columns = [str(col).strip().lower() for col in df.columns]
    required = {"timestamp", "high", "low", "close"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {file_path.name}: {', '.join(missing)}")

    out = df[[col for col in ["timestamp", "high", "low", "close", "symbol"] if col in df.columns]].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    if out["timestamp"].isna().any():
        raise ValueError(f"Found unparsable timestamps in {file_path.name}")

    if out["timestamp"].dt.tz is None:
        out["timestamp"] = out["timestamp"].dt.tz_localize(timezone)
    else:
        out["timestamp"] = out["timestamp"].dt.tz_convert(timezone)

    for column in ("high", "low", "close"):
        out[column] = pd.to_numeric(out[column], errors="coerce")

    if "symbol" not in out.columns:
        out["symbol"] = file_path.name.split("_")[0].upper()

    out = out.dropna(subset=["timestamp", "high", "low", "close"]).copy()
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    return out


def prepare_rth_data(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only RTH bars and attach session-level helpers."""
    out = df.copy()
    out["minute_of_day"] = _minute_of_day(out["timestamp"])
    out = out.loc[out["minute_of_day"].between(RTH_START_MINUTE, RTH_END_MINUTE)].copy()
    out["session_date"] = out["timestamp"].dt.date
    return out.reset_index(drop=True)


def evaluate_session_window(session_df: pd.DataFrame, symbol: str, or_window_minutes: int) -> dict[str, Any]:
    """Evaluate one session for one opening-range window."""
    session = session_df.sort_values("timestamp").reset_index(drop=True).copy()
    session_date = session["session_date"].iloc[0]
    timezone = str(session["timestamp"].iloc[0].tz)
    result: dict[str, Any] = {
        "asset": str(symbol).upper(),
        "session_date": pd.Timestamp(session_date),
        "or_window_minutes": int(or_window_minutes),
        "eligible": False,
        "exclude_reason": "",
        "breakout_direction": "no_breakout",
        "same_direction": pd.NA,
        "failed_direction": pd.NA,
        "close_extension": np.nan,
        "or_high": np.nan,
        "or_low": np.nan,
        "breakout_timestamp": pd.NaT,
        "close_reference_timestamp": pd.NaT,
        "close_reference_exact_1600": False,
        "timezone": timezone,
        "bars_in_session": int(len(session)),
    }

    or_end_minute = RTH_START_MINUTE + int(or_window_minutes)
    opening_range = session.loc[session["minute_of_day"].between(RTH_START_MINUTE, or_end_minute - 1)].copy()
    if len(opening_range) < int(or_window_minutes):
        result["exclude_reason"] = "incomplete_opening_range"
        return result

    close_proxy = session.loc[session["minute_of_day"].between(CLOSE_PROXY_MINUTE, RTH_END_MINUTE)].copy()
    if close_proxy.empty:
        result["exclude_reason"] = "no_close_proxy"
        return result

    close_bar = close_proxy.iloc[-1]
    close_timestamp = pd.Timestamp(close_bar["timestamp"])
    close_value = float(close_bar["close"])
    close_minute = int(close_bar["minute_of_day"])
    exact_close = close_minute == RTH_END_MINUTE

    or_high = float(opening_range["high"].max())
    or_low = float(opening_range["low"].min())
    post_or = session.loc[
        (session["minute_of_day"] >= or_end_minute) & (session["timestamp"] < close_timestamp)
    ].copy()

    result.update(
        {
            "eligible": True,
            "exclude_reason": "",
            "or_high": or_high,
            "or_low": or_low,
            "close_reference_timestamp": close_timestamp,
            "close_reference_exact_1600": bool(exact_close),
            "close_reference_close": close_value,
            "close_reference_minute": close_minute,
            "n_opening_range_bars": int(len(opening_range)),
            "n_post_or_bars_before_close": int(len(post_or)),
        }
    )

    breakout_candidates = post_or.loc[(post_or["close"] > or_high) | (post_or["close"] < or_low)].copy()
    if breakout_candidates.empty:
        return result

    breakout_bar = breakout_candidates.iloc[0]
    breakout_close = float(breakout_bar["close"])
    breakout_direction = "up" if breakout_close > or_high else "down"
    same_direction = close_value > or_high if breakout_direction == "up" else close_value < or_low
    extension = close_value - or_high if breakout_direction == "up" else or_low - close_value

    result.update(
        {
            "breakout_direction": breakout_direction,
            "same_direction": bool(same_direction),
            "failed_direction": bool(not same_direction),
            "close_extension": float(extension),
            "breakout_timestamp": pd.Timestamp(breakout_bar["timestamp"]),
            "breakout_close": breakout_close,
        }
    )
    return result


def summarize_breakout_results(day_level: pd.DataFrame) -> pd.DataFrame:
    """Aggregate day-level breakout outcomes by asset and OR window."""
    eligible = day_level.loc[day_level["eligible"]].copy()
    if eligible.empty:
        columns = [
            "asset",
            "or_window_minutes",
            "n_days",
            "n_no_breakout",
            "n_breakout_up",
            "n_breakout_down",
            "n_valid_breakouts",
            "n_same_direction",
            "n_failed_direction",
            "pct_no_breakout",
            "pct_breakout_up",
            "pct_breakout_down",
            "hit_rate",
            "hit_rate_up",
            "hit_rate_down",
            "avg_close_extension",
            "median_close_extension",
        ]
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    grouped = eligible.groupby(["asset", "or_window_minutes"], sort=True)
    for (asset, window), subset in grouped:
        breakout_up = subset["breakout_direction"].eq("up")
        breakout_down = subset["breakout_direction"].eq("down")
        valid = breakout_up | breakout_down
        same_direction = subset["same_direction"].eq(True)
        failed_direction = subset["failed_direction"].eq(True)
        extensions = pd.to_numeric(subset.loc[valid, "close_extension"], errors="coerce").dropna()

        n_days = int(len(subset))
        n_breakout_up = int(breakout_up.sum())
        n_breakout_down = int(breakout_down.sum())
        n_valid = int(valid.sum())
        n_same = int(same_direction.sum())
        n_failed = int(failed_direction.sum())
        n_same_up = int((breakout_up & same_direction).sum())
        n_same_down = int((breakout_down & same_direction).sum())
        n_no_breakout = int(subset["breakout_direction"].eq("no_breakout").sum())

        rows.append(
            {
                "asset": asset,
                "or_window_minutes": int(window),
                "n_days": n_days,
                "n_no_breakout": n_no_breakout,
                "n_breakout_up": n_breakout_up,
                "n_breakout_down": n_breakout_down,
                "n_valid_breakouts": n_valid,
                "n_same_direction": n_same,
                "n_failed_direction": n_failed,
                "pct_no_breakout": _safe_div(n_no_breakout, n_days),
                "pct_breakout_up": _safe_div(n_breakout_up, n_days),
                "pct_breakout_down": _safe_div(n_breakout_down, n_days),
                "hit_rate": _safe_div(n_same, n_valid),
                "hit_rate_up": _safe_div(n_same_up, n_breakout_up),
                "hit_rate_down": _safe_div(n_same_down, n_breakout_down),
                "avg_close_extension": float(extensions.mean()) if not extensions.empty else np.nan,
                "median_close_extension": float(extensions.median()) if not extensions.empty else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["asset", "or_window_minutes"]).reset_index(drop=True)
    return out


def build_data_sanity(day_level: pd.DataFrame, asset_metadata: pd.DataFrame) -> pd.DataFrame:
    """Create a compact data-sanity table by asset and OR window."""
    rows: list[dict[str, Any]] = []
    for (asset, window), subset in day_level.groupby(["asset", "or_window_minutes"], sort=True):
        meta = asset_metadata.loc[asset_metadata["asset"].eq(asset)].iloc[0]
        eligible = subset["eligible"].astype(bool)
        rows.append(
            {
                "asset": asset,
                "or_window_minutes": int(window),
                "dataset_path": meta["dataset_path"],
                "first_timestamp": meta["first_timestamp"],
                "last_timestamp": meta["last_timestamp"],
                "timezone": meta["timezone"],
                "n_rth_sessions_total": int(meta["n_rth_sessions_total"]),
                "median_rth_bars": float(meta["median_rth_bars"]),
                "n_sessions_exact_1600": int(meta["n_sessions_exact_1600"]),
                "n_sessions_close_proxy_1555_plus": int(meta["n_sessions_close_proxy_1555_plus"]),
                "n_complete_opening_range": int(subset["exclude_reason"].ne("incomplete_opening_range").sum()),
                "n_eligible_days": int(eligible.sum()),
                "n_excluded_incomplete_opening_range": int(subset["exclude_reason"].eq("incomplete_opening_range").sum()),
                "n_excluded_no_close_proxy": int(subset["exclude_reason"].eq("no_close_proxy").sum()),
                "pct_eligible_days": _safe_div(int(eligible.sum()), int(len(subset))),
            }
        )

    return pd.DataFrame(rows).sort_values(["asset", "or_window_minutes"]).reset_index(drop=True)


def _metric_pivot(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        results.pivot(index="asset", columns="or_window_minutes", values=metric)
        .reindex(index=list(DEFAULT_SYMBOLS), columns=list(DEFAULT_OR_WINDOWS))
        .copy()
    )


def _format_annotation(value: float, style: str) -> str:
    if pd.isna(value):
        return ""
    if style == "pct":
        return f"{float(value):.1%}"
    if style == "int":
        return f"{int(round(float(value))):,}"
    return f"{float(value):.2f}"


def plot_metric_heatmap(
    results: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
    cmap: str,
    style: str,
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
) -> Path:
    pivot = _metric_pivot(results, metric)
    annotations = pivot.apply(lambda column: column.map(lambda value: _format_annotation(value, style)))

    plt.figure(figsize=(8.8, 3.8))
    sns.set_theme(style="white")
    ax = sns.heatmap(
        pivot,
        annot=annotations,
        fmt="",
        cmap=cmap,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"shrink": 0.82},
        vmin=vmin,
        vmax=vmax,
        center=center,
    )
    ax.set_title(title, fontsize=12, pad=12)
    ax.set_xlabel("OR window (minutes)")
    ax.set_ylabel("Asset")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def aggregate_asset_overview(day_level: pd.DataFrame) -> pd.DataFrame:
    """Aggregate valid breakouts across all OR windows for summary answers."""
    eligible = day_level.loc[day_level["eligible"]].copy()
    valid = eligible.loc[eligible["breakout_direction"].isin(BREAKOUT_DIRECTIONS)].copy()
    rows: list[dict[str, Any]] = []

    for asset in DEFAULT_SYMBOLS:
        eligible_asset = eligible.loc[eligible["asset"].eq(asset)].copy()
        valid_asset = valid.loc[valid["asset"].eq(asset)].copy()
        up = valid_asset["breakout_direction"].eq("up")
        down = valid_asset["breakout_direction"].eq("down")
        same = valid_asset["same_direction"].eq(True)
        extensions = pd.to_numeric(valid_asset["close_extension"], errors="coerce").dropna()

        rows.append(
            {
                "asset": asset,
                "n_days": int(len(eligible_asset)),
                "n_no_breakout": int(eligible_asset["breakout_direction"].eq("no_breakout").sum()),
                "n_valid_breakouts": int(len(valid_asset)),
                "n_breakout_up": int(up.sum()),
                "n_breakout_down": int(down.sum()),
                "n_same_direction": int(same.sum()),
                "global_hit_rate": _safe_div(int(same.sum()), int(len(valid_asset))),
                "global_hit_rate_up": _safe_div(int((up & same).sum()), int(up.sum())),
                "global_hit_rate_down": _safe_div(int((down & same).sum()), int(down.sum())),
                "global_no_breakout_rate": _safe_div(
                    int(eligible_asset["breakout_direction"].eq("no_breakout").sum()),
                    int(len(eligible_asset)),
                ),
                "avg_close_extension": float(extensions.mean()) if not extensions.empty else np.nan,
                "median_close_extension": float(extensions.median()) if not extensions.empty else np.nan,
            }
        )

    return pd.DataFrame(rows).sort_values("asset").reset_index(drop=True)


def build_direct_answers(results: pd.DataFrame, day_level: pd.DataFrame) -> dict[str, Any]:
    """Compute the direct readout requested in the brief."""
    asset_overview = aggregate_asset_overview(day_level)

    best_global = asset_overview.sort_values(
        ["global_hit_rate", "n_valid_breakouts", "median_close_extension"],
        ascending=[False, False, False],
    ).iloc[0]
    best_up = asset_overview.loc[asset_overview["n_breakout_up"] > 0].sort_values(
        ["global_hit_rate_up", "n_breakout_up", "median_close_extension"],
        ascending=[False, False, False],
    ).iloc[0]
    best_down = asset_overview.loc[asset_overview["n_breakout_down"] > 0].sort_values(
        ["global_hit_rate_down", "n_breakout_down", "median_close_extension"],
        ascending=[False, False, False],
    ).iloc[0]

    best_window_by_asset = (
        results.sort_values(
            ["asset", "hit_rate", "n_valid_breakouts", "avg_close_extension"],
            ascending=[True, False, False, False],
        )
        .drop_duplicates(subset=["asset"], keep="first")
        .sort_values("asset")
        .reset_index(drop=True)
    )
    most_breakouts_by_asset = (
        results.sort_values(
            ["asset", "n_valid_breakouts", "hit_rate", "avg_close_extension"],
            ascending=[True, False, False, False],
        )
        .drop_duplicates(subset=["asset"], keep="first")
        .sort_values("asset")
        .reset_index(drop=True)
    )

    min_sample = float(results["n_valid_breakouts"].median())
    promising_pool = results.loc[results["n_valid_breakouts"] >= min_sample].copy()
    most_promising = promising_pool.sort_values(
        ["hit_rate", "n_valid_breakouts", "pct_no_breakout", "avg_close_extension"],
        ascending=[False, False, True, False],
    ).iloc[0]
    least_noisy = asset_overview.sort_values(
        ["global_hit_rate", "median_close_extension", "global_no_breakout_rate"],
        ascending=[False, False, True],
    ).iloc[0]
    least_no_breakout = results.sort_values(
        ["pct_no_breakout", "n_valid_breakouts", "hit_rate"],
        ascending=[True, False, False],
    ).iloc[0]

    return {
        "asset_overview": asset_overview,
        "best_global": best_global.to_dict(),
        "best_up": best_up.to_dict(),
        "best_down": best_down.to_dict(),
        "best_window_by_asset": best_window_by_asset.copy(),
        "most_breakouts_by_asset": most_breakouts_by_asset.copy(),
        "most_promising": most_promising.to_dict(),
        "least_noisy": least_noisy.to_dict(),
        "least_no_breakout": least_no_breakout.to_dict(),
        "min_sample_threshold": min_sample,
    }


def render_direct_answers_markdown(answers: dict[str, Any]) -> str:
    """Turn the direct-answer bundle into a short markdown section."""
    best_global = answers["best_global"]
    best_up = answers["best_up"]
    best_down = answers["best_down"]
    best_window = answers["best_window_by_asset"]
    most_breakouts = answers["most_breakouts_by_asset"]
    most_promising = answers["most_promising"]
    least_noisy = answers["least_noisy"]
    least_no_breakout = answers["least_no_breakout"]

    lines = [
        "## Direct answers",
        "",
        (
            "1. Best global hit rate: "
            f"**{best_global['asset']}** avec {float(best_global['global_hit_rate']):.1%} "
            f"sur {int(best_global['n_valid_breakouts']):,} valid breakouts agreges."
        ),
        (
            "2. Best bullish breakout hit rate: "
            f"**{best_up['asset']}** avec {float(best_up['global_hit_rate_up']):.1%} "
            f"sur {int(best_up['n_breakout_up']):,} breakouts up."
        ),
        (
            "3. Best bearish breakout hit rate: "
            f"**{best_down['asset']}** avec {float(best_down['global_hit_rate_down']):.1%} "
            f"sur {int(best_down['n_breakout_down']):,} breakouts down."
        ),
        "4. OR window with the best hit rate for each asset:",
    ]
    for _, row in best_window.iterrows():
        lines.append(
            f"- {row['asset']}: OR {int(row['or_window_minutes'])} min "
            f"({float(row['hit_rate']):.1%}, {int(row['n_valid_breakouts']):,} valid breakouts)."
        )
    lines.append("5. OR window with the most valid breakouts for each asset:")
    for _, row in most_breakouts.iterrows():
        lines.append(
            f"- {row['asset']}: OR {int(row['or_window_minutes'])} min "
            f"({int(row['n_valid_breakouts']):,} valid breakouts, hit rate {float(row['hit_rate']):.1%})."
        )
    lines.extend(
        [
            (
                "6. Most promising asset for a simple directional ORB read: "
                f"**{most_promising['asset']}** avec l'OR {int(most_promising['or_window_minutes'])} min "
                f"({float(most_promising['hit_rate']):.1%} de hit rate, "
                f"{int(most_promising['n_valid_breakouts']):,} breakouts valides, "
                f"{float(most_promising['pct_no_breakout']):.1%} de no-breakout)."
            ),
            (
                "7. Least noisy asset under this measure: "
                f"**{least_noisy['asset']}** avec {float(least_noisy['global_hit_rate']):.1%} "
                f"de hit rate global et une extension mediane de {float(least_noisy['median_close_extension']):.2f}."
            ),
            (
                "8. Asset with the fewest no-breakout days: "
                f"**{least_no_breakout['asset']}** sur l'OR {int(least_no_breakout['or_window_minutes'])} min "
                f"({float(least_no_breakout['pct_no_breakout']):.1%} de no-breakout)."
            ),
        ]
    )
    return "\n".join(lines)


def write_summary_markdown(
    results: pd.DataFrame,
    day_level: pd.DataFrame,
    data_sanity: pd.DataFrame,
    output_path: Path,
) -> Path:
    answers = build_direct_answers(results, day_level)
    asset_overview = answers["asset_overview"].copy()
    asset_overview["global_hit_rate"] = asset_overview["global_hit_rate"].map(lambda value: f"{value:.1%}")
    asset_overview["global_hit_rate_up"] = asset_overview["global_hit_rate_up"].map(lambda value: f"{value:.1%}")
    asset_overview["global_hit_rate_down"] = asset_overview["global_hit_rate_down"].map(lambda value: f"{value:.1%}")
    asset_overview["global_no_breakout_rate"] = asset_overview["global_no_breakout_rate"].map(
        lambda value: f"{value:.1%}"
    )

    sanity_lines = []
    for asset in DEFAULT_SYMBOLS:
        subset = data_sanity.loc[data_sanity["asset"].eq(asset)].copy()
        if subset.empty:
            continue
        base = subset.iloc[0]
        eligible_parts = ", ".join(
            [
                f"OR {int(row['or_window_minutes'])}m: {int(row['n_eligible_days']):,}"
                for _, row in subset.iterrows()
            ]
        )
        sanity_lines.extend(
            [
                f"- {asset}: fichier `{base['dataset_path']}`, timezone `{base['timezone']}`, "
                f"{int(base['n_rth_sessions_total']):,} sessions RTH, "
                f"{int(base['n_sessions_close_proxy_1555_plus']):,} sessions avec close proxy >= 15:55.",
                f"  Jours eligibles par fenetre: {eligible_parts}.",
            ]
        )

    top_rows = results.sort_values(
        ["hit_rate", "n_valid_breakouts", "avg_close_extension"],
        ascending=[False, False, False],
    ).head(8)
    top_table = top_rows[
        [
            "asset",
            "or_window_minutes",
            "n_days",
            "n_valid_breakouts",
            "hit_rate",
            "hit_rate_up",
            "hit_rate_down",
            "pct_no_breakout",
            "avg_close_extension",
            "median_close_extension",
        ]
    ].copy()
    for column in ["hit_rate", "hit_rate_up", "hit_rate_down", "pct_no_breakout"]:
        top_table[column] = top_table[column].map(lambda value: f"{value:.1%}")
    top_table["avg_close_extension"] = top_table["avg_close_extension"].map(lambda value: f"{value:.2f}")
    top_table["median_close_extension"] = top_table["median_close_extension"].map(lambda value: f"{value:.2f}")

    lines = [
        "# OR Window Breakout Close Direction Campaign",
        "",
        "Regle d'eligibilite retenue: journee conservee uniquement si l'Opening Range est complete pour la fenetre testee "
        "et si la session possede un close de reference coherent a `16:00` ou, a defaut, un dernier bar RTH entre `15:55` et `16:00`.",
        "La cassure est detectee sur la premiere bougie post-OR dont le `close` sort de l'OR, en arretant la recherche avant le bar utilise comme close final.",
        "",
        "## Data sanity",
        "",
        *sanity_lines,
        "",
        render_direct_answers_markdown(answers),
        "",
        "## Vue agregee par asset",
        "",
        "```text",
        asset_overview.to_string(index=False),
        "```",
        "",
        "## Top configurations",
        "",
        "```text",
        top_table.to_string(index=False),
        "```",
        "",
        "## Charts",
        "",
        "- `charts/heatmap_hit_rate.png`",
        "- `charts/heatmap_hit_rate_up.png`",
        "- `charts/heatmap_hit_rate_down.png`",
        "- `charts/heatmap_no_breakout_rate.png`",
        "- `charts/heatmap_avg_close_extension.png`",
        "- `charts/heatmap_valid_breakouts.png`",
    ]

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output_path


def build_notebook(output_root: Path) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    output_root_str = output_root.as_posix()
    nb.cells = [
        nbf.v4.new_markdown_cell("# OR Window Breakout Close Direction Validation"),
        nbf.v4.new_code_cell(
            """import sys
from pathlib import Path

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from IPython.display import Image, Markdown, display

from src.analytics.or_window_breakout_close_direction_campaign import (
    build_direct_answers,
    render_direct_answers_markdown,
)

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 240)
"""
        ),
        nbf.v4.new_markdown_cell("## 1) Charger les exports"),
        nbf.v4.new_code_cell(
            f"""OUTPUT_ROOT = Path(r"{output_root_str}")
RESULTS_PATH = OUTPUT_ROOT / "breakout_close_direction_results.csv"
DAY_LEVEL_PATH = OUTPUT_ROOT / "breakout_close_direction_day_level.csv"
SANITY_PATH = OUTPUT_ROOT / "breakout_close_direction_data_sanity.csv"
SUMMARY_PATH = OUTPUT_ROOT / "breakout_close_direction_summary.md"
CHARTS_DIR = OUTPUT_ROOT / "charts"

results = pd.read_csv(RESULTS_PATH)
day_level = pd.read_csv(DAY_LEVEL_PATH, parse_dates=["session_date", "breakout_timestamp", "close_reference_timestamp"])
data_sanity = pd.read_csv(SANITY_PATH, parse_dates=["first_timestamp", "last_timestamp"])

print("OUTPUT_ROOT =", OUTPUT_ROOT)
print("RESULTS_PATH =", RESULTS_PATH)
print("DAY_LEVEL_PATH =", DAY_LEVEL_PATH)
print("SANITY_PATH =", SANITY_PATH)
print("SUMMARY_PATH =", SUMMARY_PATH)
"""
        ),
        nbf.v4.new_markdown_cell("## 2) Tableaux recapitulatifs"),
        nbf.v4.new_code_cell(
            """display(Markdown("### Data sanity"))
display(data_sanity)

summary_cols = [
    "asset",
    "or_window_minutes",
    "n_days",
    "n_valid_breakouts",
    "hit_rate",
    "hit_rate_up",
    "hit_rate_down",
    "pct_no_breakout",
    "avg_close_extension",
    "median_close_extension",
]
display(Markdown("### Resultats detailles"))
display(results[summary_cols].sort_values(["asset", "or_window_minutes"]))

for metric in ["hit_rate", "hit_rate_up", "hit_rate_down", "pct_no_breakout", "n_valid_breakouts", "avg_close_extension"]:
    pivot = (
        results.pivot(index="asset", columns="or_window_minutes", values=metric)
        .reindex(index=["MES", "MNQ", "MGC", "M2K"], columns=[5, 15, 30, 60])
    )
    display(Markdown(f"### Pivot `{metric}`"))
    display(pivot)
"""
        ),
        nbf.v4.new_markdown_cell("## 3) Heatmaps"),
        nbf.v4.new_code_cell(
            """chart_names = [
    "heatmap_hit_rate.png",
    "heatmap_hit_rate_up.png",
    "heatmap_hit_rate_down.png",
    "heatmap_no_breakout_rate.png",
    "heatmap_avg_close_extension.png",
    "heatmap_valid_breakouts.png",
]

for name in chart_names:
    display(Markdown(f"### {name}"))
    display(Image(filename=str(CHARTS_DIR / name)))
"""
        ),
        nbf.v4.new_markdown_cell("## 4) Conclusion simple"),
        nbf.v4.new_code_cell(
            """answers = build_direct_answers(results, day_level)
display(Markdown(render_direct_answers_markdown(answers)))
"""
        ),
    ]
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "version": "3.10"}
    return nb


def write_notebook(output_root: Path, notebook_path: Path) -> Path:
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook(output_root)
    with notebook_path.open("w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)
    return notebook_path


def analyze_symbol(symbol: str, or_windows: tuple[int, ...]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one asset and compute day-level rows for every OR window."""
    dataset_path = resolve_processed_dataset(symbol)
    raw = load_processed_minute_data(dataset_path)
    rth = prepare_rth_data(raw)

    day_rows: list[dict[str, Any]] = []
    grouped = rth.groupby("session_date", sort=True)
    for _, session in grouped:
        for or_window in or_windows:
            day_rows.append(evaluate_session_window(session, symbol=symbol, or_window_minutes=int(or_window)))

    last_bar_minutes = grouped["minute_of_day"].max()
    timezone = str(rth["timestamp"].iloc[0].tz) if not rth.empty else DEFAULT_TIMEZONE
    metadata = {
        "asset": str(symbol).upper(),
        "dataset_path": dataset_path.name,
        "first_timestamp": raw["timestamp"].min(),
        "last_timestamp": raw["timestamp"].max(),
        "timezone": timezone,
        "n_rth_sessions_total": int(grouped.ngroups),
        "median_rth_bars": float(grouped.size().median()) if grouped.ngroups else 0.0,
        "n_sessions_exact_1600": int((last_bar_minutes == RTH_END_MINUTE).sum()),
        "n_sessions_close_proxy_1555_plus": int((last_bar_minutes >= CLOSE_PROXY_MINUTE).sum()),
    }
    return pd.DataFrame(day_rows), metadata


def export_results(
    results: pd.DataFrame,
    day_level: pd.DataFrame,
    data_sanity: pd.DataFrame,
    output_root: Path,
) -> dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    charts_dir = output_root / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_root / "breakout_close_direction_results.csv"
    day_level_path = output_root / "breakout_close_direction_day_level.csv"
    data_sanity_path = output_root / "breakout_close_direction_data_sanity.csv"
    summary_path = output_root / "breakout_close_direction_summary.md"

    results.to_csv(results_path, index=False)
    day_level.to_csv(day_level_path, index=False)
    data_sanity.to_csv(data_sanity_path, index=False)
    write_summary_markdown(results, day_level, data_sanity, summary_path)

    max_abs_extension = pd.to_numeric(results["avg_close_extension"], errors="coerce").abs().max()
    extension_bound = float(max_abs_extension) if math.isfinite(float(max_abs_extension)) else 1.0
    extension_bound = max(extension_bound, 1.0)

    plot_metric_heatmap(
        results,
        metric="hit_rate",
        title="Hit Rate by Asset and OR Window",
        output_path=charts_dir / "heatmap_hit_rate.png",
        cmap="RdYlGn",
        style="pct",
        vmin=0.0,
        vmax=1.0,
    )
    plot_metric_heatmap(
        results,
        metric="hit_rate_up",
        title="Hit Rate Up by Asset and OR Window",
        output_path=charts_dir / "heatmap_hit_rate_up.png",
        cmap="RdYlGn",
        style="pct",
        vmin=0.0,
        vmax=1.0,
    )
    plot_metric_heatmap(
        results,
        metric="hit_rate_down",
        title="Hit Rate Down by Asset and OR Window",
        output_path=charts_dir / "heatmap_hit_rate_down.png",
        cmap="RdYlGn",
        style="pct",
        vmin=0.0,
        vmax=1.0,
    )
    plot_metric_heatmap(
        results,
        metric="pct_no_breakout",
        title="No Breakout Rate by Asset and OR Window",
        output_path=charts_dir / "heatmap_no_breakout_rate.png",
        cmap="RdYlGn_r",
        style="pct",
        vmin=0.0,
        vmax=1.0,
    )
    plot_metric_heatmap(
        results,
        metric="avg_close_extension",
        title="Average Close Extension by Asset and OR Window",
        output_path=charts_dir / "heatmap_avg_close_extension.png",
        cmap="RdYlGn",
        style="float",
        vmin=-extension_bound,
        vmax=extension_bound,
        center=0.0,
    )
    plot_metric_heatmap(
        results,
        metric="n_valid_breakouts",
        title="Valid Breakouts by Asset and OR Window",
        output_path=charts_dir / "heatmap_valid_breakouts.png",
        cmap="Blues",
        style="int",
        vmin=0.0,
    )

    return {
        "results": results_path,
        "day_level": day_level_path,
        "data_sanity": data_sanity_path,
        "summary": summary_path,
        "charts_dir": charts_dir,
    }


def run_campaign(config: CampaignConfig) -> dict[str, Path]:
    output_root = Path(config.output_root)
    all_day_rows: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []

    for symbol in config.symbols:
        day_level, metadata = analyze_symbol(str(symbol).upper(), config.or_windows)
        all_day_rows.append(day_level)
        metadata_rows.append(metadata)

    day_level_df = pd.concat(all_day_rows, ignore_index=True)
    metadata_df = pd.DataFrame(metadata_rows).sort_values("asset").reset_index(drop=True)
    results_df = summarize_breakout_results(day_level_df)
    data_sanity_df = build_data_sanity(day_level_df, metadata_df)
    exports = export_results(results_df, day_level_df, data_sanity_df, output_root)
    notebook_path = write_notebook(output_root, Path(config.notebook_path))
    exports["notebook"] = notebook_path
    return exports


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a simple descriptive OR-window close-breakout direction campaign."
    )
    parser.add_argument("--symbols", nargs="*", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--or-windows", nargs="*", type=int, default=list(DEFAULT_OR_WINDOWS))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--notebook-path", type=Path, default=DEFAULT_NOTEBOOK_PATH)
    args = parser.parse_args()

    config = CampaignConfig(
        symbols=tuple(str(symbol).upper() for symbol in args.symbols),
        or_windows=tuple(int(window) for window in args.or_windows),
        output_root=Path(args.output_root),
        notebook_path=Path(args.notebook_path),
    )
    exports = run_campaign(config)
    for key, path in exports.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
