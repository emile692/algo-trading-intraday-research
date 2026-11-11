"""Causal Phase-1 diagnostic for 60/40 rebalancing pressure."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.analytics.close_of_day_intraday_momentum import newey_west_regression
from src.config.settings import INSTRUMENT_SPECS


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "research" / "sixty_forty_rebalancing_20261002"
DEFAULT_SLEEVES = REPO_ROOT / "research" / "daily_pnl.csv"
TIMEZONE = "America/New_York"


@dataclass(frozen=True)
class StudyVariant:
    name: str
    equity_symbol: str
    bond_symbol: str
    reset_frequency: str = "monthly"
    trade_symbol: str | None = None


DEFAULT_VARIANTS = (
    StudyVariant("mes_zn_monthly", "MES", "ZN", "monthly", "MES"),
    StudyVariant("mes_zf_monthly", "MES", "ZF", "monthly", "MES"),
    StudyVariant("mnq_zn_monthly", "MNQ", "ZN", "monthly", "MNQ"),
    StudyVariant("nq_zn_monthly_extended", "NQ", "ZN", "monthly", None),
    StudyVariant("mes_zn_quarterly_reset", "MES", "ZN", "quarterly", "MES"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp_column(frame: pd.DataFrame) -> pd.DataFrame:
    if isinstance(frame.index, pd.DatetimeIndex):
        out = frame.reset_index()
        out = out.rename(columns={out.columns[0]: "timestamp"})
        return out
    out = frame.copy()
    if "timestamp" not in out.columns and "ts_event" in out.columns:
        out = out.rename(columns={"ts_event": "timestamp"})
    if "timestamp" not in out.columns:
        raise ValueError("Dataset has no timestamp index or column")
    return out


def load_minute_bars(path: Path) -> pd.DataFrame:
    """Load only fields needed by this study and normalize to US Eastern."""
    frame = _timestamp_column(pd.read_parquet(path))
    required = {"timestamp", "open", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    keep = [column for column in ("timestamp", "open", "close", "instrument_id") if column in frame]
    frame = frame[keep].copy()
    timestamp = pd.to_datetime(frame["timestamp"], errors="raise")
    if timestamp.dt.tz is None:
        timestamp = timestamp.dt.tz_localize("UTC")
    frame["timestamp"] = timestamp.dt.tz_convert(TIMEZONE)
    return frame.sort_values("timestamp").reset_index(drop=True)


def extract_daily_snapshots(bars: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Extract signal, entry and exit bars without using a bar that closes after entry."""
    frame = bars.copy()
    frame["session_date"] = frame["timestamp"].dt.tz_localize(None).dt.normalize()
    frame["minute"] = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
    targets = frame.loc[frame["minute"].isin((570, 899, 900, 959))].copy()
    # 09:30 open supports the next-session diagnostic. Signal is the 14:59 close;
    # execution starts at the 15:00 bar open and exits on the 15:59 bar close.
    pieces: list[pd.DataFrame] = []
    for minute, label, field in (
        (570, "rth_open", "open"),
        (899, "signal_price", "close"),
        (900, "entry_price", "open"),
        (959, "close_price", "close"),
    ):
        selected = targets.loc[targets["minute"].eq(minute), ["session_date", field] + (["instrument_id"] if "instrument_id" in targets else [])]
        selected = selected.drop_duplicates("session_date", keep="last").rename(columns={field: label})
        if "instrument_id" in selected:
            selected = selected.rename(columns={"instrument_id": f"{label}_instrument_id"})
        pieces.append(selected)
    daily = pieces[0]
    for piece in pieces[1:]:
        daily = daily.merge(piece, on="session_date", how="outer", validate="one_to_one")
    daily["symbol"] = symbol
    return daily.sort_values("session_date").reset_index(drop=True)


def add_roll_neutral_index(daily: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Create a chained index without treating a cross-contract gap as return.

    On a roll session, the overnight cross-contract move is unknowable from a
    rank-0 continuous file. We retain only the same-contract 09:30-to-snapshot
    return for that day. This avoids selectively losing common month-end roll
    dates while remaining explicit about the omitted overnight component.
    """
    out = daily.copy().sort_values("session_date").reset_index(drop=True)
    close = pd.to_numeric(out[f"{prefix}_close_price"], errors="coerce")
    instrument_col = f"{prefix}_close_price_instrument_id"
    same_contract = pd.Series(True, index=out.index)
    if instrument_col in out:
        same_contract = out[instrument_col].eq(out[instrument_col].shift())
    raw_return = close.pct_change(fill_method=None)
    rth_open = pd.to_numeric(out[f"{prefix}_rth_open"], errors="coerce")
    roll_intraday_return = close / rth_open - 1.0
    valid_return = raw_return.where(same_contract, roll_intraday_return)
    out[f"{prefix}_roll_transition"] = (~same_contract & close.notna() & close.shift().notna()).fillna(False)
    out[f"{prefix}_close_return"] = valid_return
    out[f"{prefix}_index"] = (1.0 + valid_return.fillna(0.0)).cumprod()
    signal = pd.to_numeric(out[f"{prefix}_signal_price"], errors="coerce")
    signal_instrument = f"{prefix}_signal_price_instrument_id"
    signal_same = same_contract.copy()
    if signal_instrument in out and instrument_col in out:
        signal_same &= out[signal_instrument].eq(out[instrument_col])
    regular_signal_growth = signal / close.shift()
    roll_signal_growth = signal / rth_open
    intraday = regular_signal_growth.where(same_contract, roll_signal_growth).where(signal_same | ~same_contract)
    rth_instrument = f"{prefix}_rth_open_instrument_id"
    if rth_instrument in out and signal_instrument in out and instrument_col in out:
        roll_snapshot_same_contract = out[rth_instrument].eq(out[signal_instrument]) & out[rth_instrument].eq(out[instrument_col])
        intraday = intraday.where(same_contract | roll_snapshot_same_contract)
    out[f"{prefix}_signal_index"] = out[f"{prefix}_index"].shift() * intraday
    return out


def _period_key(dates: pd.Series, frequency: str) -> pd.Series:
    if frequency == "monthly":
        return dates.dt.to_period("M")
    if frequency == "quarterly":
        return dates.dt.to_period("Q")
    raise ValueError(f"Unsupported reset frequency: {frequency}")


def compute_drift(events: pd.DataFrame, reset_frequency: str) -> pd.DataFrame:
    """Compute the observable equity-weight drift from a lagged 60/40 reset."""
    out = events.copy().sort_values("session_date").reset_index(drop=True)
    period = _period_key(out["session_date"], reset_frequency)
    out["reset_period"] = period.astype(str)
    # On the first session of a period, shift(1) is exactly the prior session close.
    equity_anchor = out["equity_index"].shift().groupby(period).transform("first")
    bond_anchor = out["bond_index"].shift().groupby(period).transform("first")
    equity_growth = out["equity_signal_index"] / equity_anchor
    bond_growth = out["bond_signal_index"] / bond_anchor
    equity_value = 0.60 * equity_growth
    bond_value = 0.40 * bond_growth
    out["equity_weight"] = equity_value / (equity_value + bond_value)
    out["drift"] = out["equity_weight"] - 0.60
    return out


def add_calendar_flags(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy().sort_values("session_date").reset_index(drop=True)
    month = out["session_date"].dt.to_period("M")
    out["session_from_month_end"] = out.groupby(month).cumcount(ascending=False) + 1
    out["is_last_week"] = out["session_from_month_end"].le(5)
    out["is_month_end"] = out["session_from_month_end"].eq(1)
    out["is_quarter_end"] = out["is_month_end"] & out["session_date"].dt.month.isin((3, 6, 9, 12))
    return out


def build_events(equity: pd.DataFrame, bond: pd.DataFrame, variant: StudyVariant) -> pd.DataFrame:
    eq = equity.add_prefix("equity_").rename(columns={"equity_session_date": "session_date"})
    bd = bond.add_prefix("bond_").rename(columns={"bond_session_date": "session_date"})
    merged = eq.merge(bd, on="session_date", how="inner", validate="one_to_one")
    merged = add_roll_neutral_index(merged, "equity")
    merged = add_roll_neutral_index(merged, "bond")
    merged = compute_drift(merged, variant.reset_frequency)
    merged = add_calendar_flags(merged)
    merged["variant"] = variant.name
    merged["future_final_hour_return"] = merged["equity_close_price"] / merged["equity_entry_price"] - 1.0
    next_session_rth = (merged["equity_close_price"] / merged["equity_rth_open"] - 1.0).shift(-1)
    ids = [column for column in ("equity_signal_price_instrument_id", "equity_entry_price_instrument_id", "equity_close_price_instrument_id") if column in merged]
    if ids:
        merged["valid_execution_contract"] = merged[ids].nunique(axis=1).eq(1)
    else:
        merged["valid_execution_contract"] = True
    merged.loc[~merged["valid_execution_contract"], "future_final_hour_return"] = np.nan
    merged["next_rth_return"] = next_session_rth.where(merged["valid_execution_contract"].shift(-1).eq(True))
    merged["contrarian_edge"] = -np.sign(merged["drift"]) * merged["future_final_hour_return"]
    return merged


def _simple_t(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(stats.ttest_1samp(clean, 0.0).statistic) if len(clean) > 1 else np.nan


def _newey_west_mean(values: pd.Series) -> tuple[float, int]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    n = len(clean)
    if n < 2:
        return np.nan, 0
    residuals = clean - clean.mean()
    lags = int(math.floor(4 * (n / 100) ** (2 / 9)))
    long_variance = float(residuals @ residuals / n)
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        covariance = float(residuals[lag:] @ residuals[:-lag] / n)
        long_variance += 2.0 * weight * covariance
    standard_error = math.sqrt(max(long_variance, 0.0) / n)
    return (float(clean.mean() / standard_error) if standard_error else np.nan), lags


def diagnostic_tables(events_by_variant: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    regression_rows: list[dict[str, Any]] = []
    conditional_rows: list[dict[str, Any]] = []
    quantile_rows: list[dict[str, Any]] = []
    for variant, events in events_by_variant.items():
        masks = {
            "all_sessions": pd.Series(True, index=events.index),
            "last_week": events["is_last_week"],
            "month_end": events["is_month_end"],
            "quarter_end": events["is_quarter_end"],
        }
        horizons = {"final_hour": "future_final_hour_return", "next_session_rth": "next_rth_return"}
        for window, mask in masks.items():
            for horizon, return_column in horizons.items():
                sample = events.loc[mask, ["drift", return_column]].dropna().rename(columns={return_column: "future_return"})
                sample["contrarian_edge"] = -np.sign(sample["drift"]) * sample["future_return"]
                regression = newey_west_regression(sample["future_return"], sample["drift"])
                regression_rows.append({
                    "variant": variant,
                    "window": window,
                    "horizon": horizon,
                    **regression,
                    "beta_bps_per_1pp_drift": regression["beta"] * 100.0 if pd.notna(regression["beta"]) else np.nan,
                })
                edge = sample["contrarian_edge"]
                hac_t, hac_lags = _newey_west_mean(edge)
                conditional_rows.append({
                    "variant": variant,
                    "window": window,
                    "horizon": horizon,
                    "observations": len(sample),
                    "mean_future_return_bps": sample["future_return"].mean() * 10_000,
                    "mean_contrarian_edge_bps": edge.mean() * 10_000,
                    "simple_t": _simple_t(edge),
                    "hac_t": hac_t,
                    "hac_lags": hac_lags,
                    "hit_rate": edge.gt(0).mean() if len(edge) else np.nan,
                    "mean_abs_drift_pp": sample["drift"].abs().mean() * 100,
                })
                if horizon == "final_hour" and len(sample) >= 10:
                    bins = min(5, sample["drift"].nunique())
                    ranked = sample.assign(quantile=pd.qcut(sample["drift"], bins, labels=False, duplicates="drop") + 1)
                    for quantile, group in ranked.groupby("quantile", observed=True):
                        quantile_rows.append({
                            "variant": variant,
                            "window": window,
                            "horizon": horizon,
                            "drift_quantile": int(quantile),
                            "observations": len(group),
                            "mean_drift_pp": group["drift"].mean() * 100,
                            "mean_future_return_bps": group["future_return"].mean() * 10_000,
                            "simple_t": _simple_t(group["future_return"]),
                        })
    return pd.DataFrame(regression_rows), pd.DataFrame(conditional_rows), pd.DataFrame(quantile_rows)


def build_trades(events: pd.DataFrame, symbol: str, window: str = "month_end", slippage_ticks: float = 1.0) -> pd.DataFrame:
    flag = {"last_week": "is_last_week", "month_end": "is_month_end", "quarter_end": "is_quarter_end"}[window]
    trades = events.loc[events[flag] & events["drift"].notna() & events["future_final_hour_return"].notna()].copy()
    trades["position"] = -np.sign(trades["drift"]).astype(int)
    spec = INSTRUMENT_SPECS[symbol]
    trades["gross_pnl_usd"] = trades["position"] * (trades["equity_close_price"] - trades["equity_entry_price"]) * float(spec["point_value_usd"])
    trades["commission_usd"] = 2.0 * float(spec["commission_per_side_usd"])
    trades["slippage_usd"] = 2.0 * slippage_ticks * float(spec["tick_value_usd"])
    trades["net_pnl_usd"] = trades["gross_pnl_usd"] - trades["commission_usd"] - trades["slippage_usd"]
    trades["symbol"] = symbol
    trades["trade_window"] = window
    trades["slippage_ticks_per_side"] = slippage_ticks
    return trades


def summarize_trades(trades: pd.DataFrame, label: str) -> dict[str, Any]:
    pnl = trades["net_pnl_usd"]
    equity = pnl.cumsum()
    drawdown = equity - equity.cummax()
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    return {
        "strategy": label,
        "trades": len(trades),
        "gross_pnl_usd": trades["gross_pnl_usd"].sum(),
        "total_cost_usd": (trades["commission_usd"] + trades["slippage_usd"]).sum(),
        "net_pnl_usd": pnl.sum(),
        "avg_gross_pnl_usd": trades["gross_pnl_usd"].mean(),
        "avg_net_pnl_usd": pnl.mean(),
        "simple_t_net": _simple_t(pnl),
        "win_rate_net": pnl.gt(0).mean(),
        "profit_factor_net": gains / losses if losses else np.inf,
        "max_drawdown_usd": drawdown.min(),
    }


def temporal_stability(trades: pd.DataFrame, strategy: str) -> pd.DataFrame:
    years = trades["session_date"].dt.year
    periods = {
        "2019-2021": years.between(2019, 2021),
        "2022-2023": years.between(2022, 2023),
        "2024-2026": years.between(2024, 2026),
    }
    rows = []
    for period, mask in periods.items():
        row = summarize_trades(trades.loc[mask], strategy)
        row["period"] = period
        rows.append(row)
    return pd.DataFrame(rows)


def sleeve_correlations(trades: pd.DataFrame, sleeves_path: Path = DEFAULT_SLEEVES) -> pd.DataFrame:
    sleeves = pd.read_csv(sleeves_path, parse_dates=["session_date"])
    strategy = trades.groupby("session_date", as_index=False)["net_pnl_usd"].sum()
    strategy["strategy_trade"] = True
    merged = sleeves.merge(strategy, on="session_date", how="left")
    merged["net_pnl_usd"] = merged["net_pnl_usd"].fillna(0.0)
    pnl_columns = [column for column in sleeves.columns if column != "session_date"]
    rows = [
        {
            "sleeve": column,
            "daily_pnl_correlation": merged["net_pnl_usd"].corr(merged[column]),
            "observations": len(merged),
            "strategy_active_days": int(merged["strategy_trade"].eq(True).sum()),
        }
        for column in pnl_columns
    ]
    merged["six_sleeve_equal_contract_pnl_usd"] = merged[pnl_columns].sum(axis=1)
    rows.append({
        "sleeve": "six_sleeve_equal_contract_pnl_usd",
        "daily_pnl_correlation": merged["net_pnl_usd"].corr(merged["six_sleeve_equal_contract_pnl_usd"]),
        "observations": len(merged),
        "strategy_active_days": int(merged["strategy_trade"].eq(True).sum()),
    })
    return pd.DataFrame(rows)


def data_inventory(paths: dict[str, Path], snapshots: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for symbol, path in paths.items():
        daily = snapshots[symbol]
        rows.append({
            "symbol": symbol,
            "path": str(path.relative_to(REPO_ROOT)),
            "file_size_mb": path.stat().st_size / 1_000_000,
            "first_session": daily["session_date"].min(),
            "last_session": daily["session_date"].max(),
            "sessions": daily["session_date"].nunique(),
            "sha256": sha256(path),
        })
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, float_digits: int = 3) -> str:
    """Render a compact Markdown table without the optional tabulate package."""
    if frame.empty:
        return "_Aucune observation._"

    def render(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{float_digits}f}"
        if isinstance(value, pd.Timestamp):
            return value.strftime("%Y-%m-%d")
        return str(value).replace("|", "\\|")

    headers = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    rows.extend("| " + " | ".join(render(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def write_report(output: Path, inventory: pd.DataFrame, regressions: pd.DataFrame, conditional: pd.DataFrame,
                 quantiles: pd.DataFrame, backtests: pd.DataFrame, stability: pd.DataFrame,
                 correlations: pd.DataFrame) -> None:
    central_reg = regressions[(regressions["variant"] == "mes_zn_monthly") & regressions["window"].isin(("last_week", "month_end", "quarter_end")) & regressions["horizon"].eq("final_hour")]
    central_cond = conditional[(conditional["variant"] == "mes_zn_monthly") & conditional["window"].isin(("last_week", "month_end", "quarter_end")) & conditional["horizon"].eq("final_hour")]
    central_quantiles = quantiles[(quantiles["variant"] == "mes_zn_monthly") & quantiles["window"].eq("last_week") & quantiles["horizon"].eq("final_hour")]
    horizon_check = conditional[(conditional["variant"] == "mes_zn_monthly") & conditional["window"].isin(("last_week", "month_end", "quarter_end"))]
    variant_check = conditional[conditional["window"].eq("month_end") & conditional["horizon"].eq("final_hour")]
    central_bt = backtests.loc[backtests["strategy"].eq("MES month-end, 1 tick/side")].iloc[0]
    slope = central_reg.loc[central_reg["window"].eq("month_end"), "hac_t_stat"].iloc[0]
    edge_t = central_cond.loc[central_cond["window"].eq("month_end"), "hac_t"].iloc[0]
    central_stability = stability[stability["strategy"].eq("MES month-end, 1 tick/side")]
    positive_periods = int(central_stability["net_pnl_usd"].gt(0).sum())
    credible = bool(slope < -1.0 and edge_t > 1.0 and central_bt["net_pnl_usd"] > 0 and positive_periods >= 2)
    verdict = "OUI — signal assez crédible pour une Phase 2" if credible else "NON — preuve Phase 1 insuffisante pour justifier une Phase 2"
    lines = [
        "# Flux de rééquilibrage 60/40 — diagnostic Phase 1",
        "",
        f"## Verdict : {verdict}",
        "",
        "Le verdict est mécaniste et volontairement exigeant : pente month-end négative, edge contrarian positif, PnL net positif et au moins deux sous-périodes positives. Il ne constitue pas une validation statistique définitive.",
        "",
        "## Méthode causale",
        "",
        "- Central : portefeuille théorique 60% MES / 40% ZN, remis à 60/40 à la clôture du mois précédent.",
        "- Signal : dérive du poids actions calculée avec la clôture de la barre 14:59 ET; aucune donnée postérieure à 15:00 n'entre dans le signal.",
        "- Rendement futur : ouverture de la barre 15:00 jusqu'à la clôture de la barre 15:59 ET.",
        "- Les sauts entre contrats continus sont exclus de la chaîne : lors d'un roll, seul le rendement même contrat 09:30-snapshot est conservé; le gap overnight inter-contrats est neutralisé.",
        "- Fenêtres calendaires connues ex ante : cinq dernières séances, dernière séance du mois, fin de trimestre.",
        "",
        "## Inventaire des données",
        "",
        _markdown_table(inventory),
        "",
        "## Diagnostic économique central",
        "",
        "### Régression rendement futur sur dérive",
        "",
        _markdown_table(central_reg[["window", "n", "beta_bps_per_1pp_drift", "t_stat", "hac_t_stat", "r_squared"]]),
        "",
        "Une pente négative va dans le sens attendu : plus le 60/40 est surpondéré en actions, plus le rendement actions suivant devrait être faible.",
        "",
        "### Rendement contrarian conditionnel",
        "",
        _markdown_table(central_cond[["window", "observations", "mean_contrarian_edge_bps", "simple_t", "hac_t", "hit_rate", "mean_abs_drift_pp"]]),
        "",
        "### Monotonie — quintiles de dérive, dernière semaine",
        "",
        _markdown_table(central_quantiles[["drift_quantile", "observations", "mean_drift_pp", "mean_future_return_bps", "simple_t"]]),
        "",
        "### SensibilitÃ© d'horizon et variantes Ã©conomiques",
        "",
        _markdown_table(horizon_check[["window", "horizon", "observations", "mean_contrarian_edge_bps", "hac_t", "hit_rate"]]),
        "",
        _markdown_table(variant_check[["variant", "observations", "mean_contrarian_edge_bps", "hac_t", "hit_rate", "mean_abs_drift_pp"]]),
        "",
        "Le contrÃ´le `next_session_rth` demande si la dÃ©rive prÃ©dit la sÃ©ance RTH suivante; il ne sert pas Ã  sÃ©lectionner la rÃ¨gle. Les variantes changent un seul choix Ã©conomique Ã  la fois et ne constituent pas une grille d'optimisation.",
        "",
        "## Backtest simple",
        "",
        _markdown_table(backtests, float_digits=2),
        "",
        "Règle centrale : une position MES à 15:00 le dernier jour de bourse du mois, opposée au signe de la dérive, sortie à 16:00. Les coûts nominaux sont 2,50 $ de commissions aller-retour plus 2,50 $ de slippage (1 tick par côté), soit 5,00 $ par trade. Le stress porte le slippage à 2 ticks par côté.",
        "",
        "## Stabilité temporelle de la règle centrale",
        "",
        _markdown_table(stability[["strategy", "period", "trades", "gross_pnl_usd", "total_cost_usd", "net_pnl_usd", "avg_net_pnl_usd", "simple_t_net", "win_rate_net", "max_drawdown_usd"]], float_digits=2),
        "",
        "## Corrélation avec les six sleeves gelées",
        "",
        _markdown_table(correlations),
        "",
        "La corrélation est calculée sur le PnL journalier commun, avec zéro les jours sans trade 60/40. Les six séries existantes sont seulement lues; elles ne sont ni modifiées ni recalculées.",
        "",
        "## Limites avant Phase 2",
        "",
        "- MES ne commence qu'en mai 2019 : environ sept ans et peu de fins de trimestre.",
        "- Aucun fichier ES local n'est disponible; NQ fournit l'extension historique depuis 2010 mais n'est pas un proxy neutre du marchÃ© actions 60/40.",
        "- Un futur Treasury est un proxy imparfait du rendement total d'un portefeuille obligataire cash.",
        "- Les séries continues ne sont pas back-adjustées; neutraliser les transitions évite les faux sauts mais omet la composante overnight lors de chaque roll.",
        "- La dernière séance mensuelle est identifiée sur le calendrier observé; en production elle doit venir d'un calendrier CME préchargé.",
        "- Cette Phase 1 ne réalise volontairement ni DSR, ni bootstrap complexe, ni ERC, ni simulation Topstep.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_study(paths: dict[str, Path], output: Path = DEFAULT_OUTPUT) -> dict[str, pd.DataFrame]:
    output.mkdir(parents=True, exist_ok=True)
    snapshots = {symbol: extract_daily_snapshots(load_minute_bars(path), symbol) for symbol, path in paths.items()}
    variants = [variant for variant in DEFAULT_VARIANTS if variant.equity_symbol in paths and variant.bond_symbol in paths]
    events_by_variant = {
        variant.name: build_events(snapshots[variant.equity_symbol], snapshots[variant.bond_symbol], variant)
        for variant in variants
    }
    inventory = data_inventory(paths, snapshots)
    regressions, conditional, quantiles = diagnostic_tables(events_by_variant)
    central_events = events_by_variant["mes_zn_monthly"]
    central_trades = build_trades(central_events, "MES", "month_end", slippage_ticks=1.0)
    stressed_trades = build_trades(central_events, "MES", "month_end", slippage_ticks=2.0)
    quarter_trades = build_trades(central_events, "MES", "quarter_end", slippage_ticks=1.0)
    mnq_trades = build_trades(events_by_variant["mnq_zn_monthly"], "MNQ", "month_end", slippage_ticks=1.0)
    backtests = pd.DataFrame([
        summarize_trades(central_trades, "MES month-end, 1 tick/side"),
        summarize_trades(stressed_trades, "MES month-end, 2 ticks/side"),
        summarize_trades(quarter_trades, "MES quarter-end, 1 tick/side"),
        summarize_trades(mnq_trades, "MNQ month-end, 1 tick/side"),
    ])
    stability = pd.concat([
        temporal_stability(central_trades, "MES month-end, 1 tick/side"),
        temporal_stability(mnq_trades, "MNQ month-end, 1 tick/side"),
    ], ignore_index=True)
    correlations = sleeve_correlations(central_trades)
    artifacts = {
        "data_inventory": inventory,
        "diagnostic_regressions": regressions,
        "conditional_returns": conditional,
        "drift_quantiles": quantiles,
        "backtest_summary": backtests,
        "temporal_stability": stability,
        "sleeve_correlations": correlations,
        "central_trades": central_trades,
    }
    for name, frame in artifacts.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    for name, frame in events_by_variant.items():
        columns = ["session_date", "variant", "equity_weight", "drift", "future_final_hour_return", "next_rth_return", "is_last_week", "is_month_end", "is_quarter_end", "valid_execution_contract"]
        frame[columns].to_csv(output / f"events_{name}.csv", index=False)
    metadata = {
        "run_date": "2026-10-03",
        "timezone": TIMEZONE,
        "signal_bar": "14:59 close",
        "entry": "15:00 open",
        "exit": "15:59 close",
        "variants": [variant.__dict__ for variant in variants],
        "input_sha256": {symbol: sha256(path) for symbol, path in paths.items()},
        "six_sleeves_path": str(DEFAULT_SLEEVES.relative_to(REPO_ROOT)),
        "six_sleeves_sha256": sha256(DEFAULT_SLEEVES),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(output, inventory, regressions, conditional, quantiles, backtests, stability, correlations)
    return artifacts

