"""Refresh the four final ORB notebooks with shared notebook helpers."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.orb_multi_asset_notebooks import NOTEBOOKS_DIR, write_notebook


MNQ_NOTEBOOK = NOTEBOOKS_DIR / "orb_MNQ_final_ensemble_validation.ipynb"

HELPER_IMPORT = """from src.analytics.orb_notebook_utils import (
    build_scope_readout_markdown,
    curve_annualized_return,
    curve_daily_sharpe,
    curve_daily_vol,
    curve_drawdown_pct,
    curve_max_drawdown_pct,
    curve_total_return_pct,
    format_curve_stats_line,
    normalize_curve,
)"""

MNQ_HELPER_BLOCK = """def _to_naive_utc(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors='coerce')
    return ts.dt.tz_convert(None)


initial_capital = float(BASELINE['account_size_usd'])


def drawdown_pct_from_equity(eq: pd.Series) -> pd.Series:
    return curve_drawdown_pct(eq)


def sharpe_daily_from_equity(df: pd.DataFrame) -> float:
    return curve_daily_sharpe(df)


def annualized_return_from_equity(df: pd.DataFrame) -> float:
    return curve_annualized_return(df, initial_capital)


def vol_daily_from_equity(df: pd.DataFrame) -> float:
    return curve_daily_vol(df)


def total_return_pct(df: pd.DataFrame) -> float:
    return curve_total_return_pct(df, initial_capital)


def max_drawdown_pct(df: pd.DataFrame) -> float:
    return curve_max_drawdown_pct(df)


def format_stats_line(name: str, sharpe: float, ret_pct: float, cagr_pct: float, vol_pct: float, dd_pct: float, pf: float | None = None, exp: float | None = None) -> str:
    return format_curve_stats_line(
        name=name,
        sharpe=sharpe,
        ret_pct=ret_pct,
        cagr_pct=cagr_pct,
        vol_pct=vol_pct,
        dd_pct=dd_pct,
        pf=pf,
        exp=exp,
    )


# Run ensemble backtest with explicit execution settings"""


def _patch_mnq_notebook(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))

    import_src = "".join(nb["cells"][1]["source"])
    if "from src.analytics.orb_notebook_utils import (" not in import_src:
        import_src = import_src.replace(
            "from src.analytics.metrics import compute_metrics\n",
            "from src.analytics.metrics import compute_metrics\n" + HELPER_IMPORT + "\n",
        )
        nb["cells"][1]["source"] = import_src.splitlines(keepends=True)

    cell_src = "".join(nb["cells"][7]["source"])
    cell_src = re.sub(
        r"def _to_naive_utc\(series: pd\.Series\) -> pd\.Series:\n.*?# Run ensemble backtest with explicit execution settings",
        MNQ_HELPER_BLOCK,
        cell_src,
        count=1,
        flags=re.DOTALL,
    )

    cell_src = cell_src.replace(
        """ensemble_eq = build_equity_curve(ensemble_trades, initial_capital=float(BASELINE['account_size_usd']))
if not ensemble_eq.empty:
    ensemble_eq['timestamp'] = _to_naive_utc(ensemble_eq['timestamp'])
    ensemble_eq = ensemble_eq.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    ensemble_eq['drawdown_pct'] = drawdown_pct_from_equity(ensemble_eq['equity'])
else:
    ensemble_eq = pd.DataFrame(columns=['timestamp','equity','drawdown','drawdown_pct'])
""",
        """ensemble_eq = normalize_curve(
    build_equity_curve(ensemble_trades, initial_capital=float(BASELINE['account_size_usd']))
)
""",
    )

    cell_src = cell_src.replace(
        """bench = pd.DataFrame({
    'timestamp': pd.to_datetime(daily_close.index),
    'equity': float(BASELINE['account_size_usd']) * (daily_close / daily_close.iloc[0]),
}).sort_values('timestamp').reset_index(drop=True)
bench['drawdown_pct'] = drawdown_pct_from_equity(bench['equity'])
""",
        """bench = normalize_curve(pd.DataFrame({
    'timestamp': pd.to_datetime(daily_close.index),
    'equity': float(BASELINE['account_size_usd']) * (daily_close / daily_close.iloc[0]),
}).sort_values('timestamp').reset_index(drop=True))
""",
    )

    scope_marker = "print('buy_hold_total_return_pct =', total_return_pct(bench))\n"
    scope_block = """print('buy_hold_total_return_pct =', total_return_pct(bench))

display(Markdown(build_scope_readout_markdown(
    full_curve=ensemble_eq,
    oos_curve=normalize_curve(build_equity_curve(
        ensemble_trades.loc[ensemble_trades['session_date'].isin(set(oos_sessions))].copy(),
        initial_capital=float(BASELINE['account_size_usd']),
    )),
    initial_capital=float(BASELINE['account_size_usd']),
    full_label='Full-sample ensemble curve',
    oos_label='OOS-only ensemble curve',
)))
"""
    if "build_scope_readout_markdown(" not in cell_src:
        cell_src = cell_src.replace(scope_marker, scope_block)

    nb["cells"][7]["source"] = cell_src.splitlines(keepends=True)
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


def main() -> None:
    for symbol in ("MES", "M2K", "MGC"):
        output = NOTEBOOKS_DIR / f"orb_{symbol}_final_ensemble_validation.ipynb"
        write_notebook(symbol, output)
        print(f"refreshed {output}")

    _patch_mnq_notebook(MNQ_NOTEBOOK)
    print(f"patched {MNQ_NOTEBOOK}")


if __name__ == "__main__":
    main()
