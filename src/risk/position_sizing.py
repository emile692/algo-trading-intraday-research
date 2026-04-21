"""Reusable fixed-contract and risk-based position sizing helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FixedContractPositionSizing:
    """Always trade a fixed number of contracts."""

    fixed_contracts: int = 1


@dataclass(frozen=True)
class RiskPercentPositionSizing:
    """Size each trade from a fixed fraction of current equity."""

    initial_capital_usd: float
    risk_pct: float
    max_contracts: int
    skip_trade_if_too_small: bool = True
    compound_realized_pnl: bool = True


PositionSizingConfig = FixedContractPositionSizing | RiskPercentPositionSizing


@dataclass(frozen=True)
class PositionSizingDecision:
    """Resolved contract count and audit fields for a trade attempt."""

    position_sizing_mode: str
    contracts: int
    contracts_raw: float | None
    capital_before_trade_usd: float | None
    risk_pct: float | None
    risk_budget_usd: float | None
    stop_distance_points: float | None
    risk_per_contract_usd: float | None
    actual_risk_usd: float | None
    notional_usd: float | None
    leverage_used: float | None
    max_contracts: int | None
    skip_trade_if_too_small: bool | None
    skipped: bool
    skip_reason: str | None


def validate_position_sizing(config: PositionSizingConfig | None) -> None:
    """Validate a sizing configuration."""
    if config is None:
        return

    if isinstance(config, FixedContractPositionSizing):
        if int(config.fixed_contracts) < 1:
            raise ValueError("fixed_contracts must be at least 1.")
        return

    if float(config.initial_capital_usd) <= 0:
        raise ValueError("initial_capital_usd must be strictly positive.")
    if float(config.risk_pct) <= 0 or float(config.risk_pct) > 1.0:
        raise ValueError("risk_pct must be in the (0, 1] interval.")
    if int(config.max_contracts) < 1:
        raise ValueError("max_contracts must be at least 1.")


def initial_capital_from_sizing(config: PositionSizingConfig | None) -> float | None:
    """Return the initial capital implied by the sizing config, when any."""
    if isinstance(config, RiskPercentPositionSizing):
        return float(config.initial_capital_usd)
    return None


def compounds_realized_pnl(config: PositionSizingConfig | None) -> bool:
    """Return whether risk-percent sizing should compound realized PnL."""
    return isinstance(config, RiskPercentPositionSizing) and bool(config.compound_realized_pnl)


def resolve_position_size(
    *,
    config: PositionSizingConfig | None,
    capital_before_trade_usd: float | None,
    entry_price: float,
    initial_stop_price: float,
    point_value_usd: float,
) -> PositionSizingDecision:
    """Return the contract count and the audit trail for a trade attempt."""
    validate_position_sizing(config)

    stop_distance_points = abs(float(entry_price) - float(initial_stop_price))
    risk_per_contract_usd = (
        stop_distance_points * float(point_value_usd)
        if math.isfinite(stop_distance_points) and math.isfinite(float(point_value_usd))
        else float("nan")
    )

    if config is None:
        config = FixedContractPositionSizing()

    if isinstance(config, FixedContractPositionSizing):
        contracts = int(config.fixed_contracts)
        notional_usd = float(contracts) * float(entry_price) * float(point_value_usd)
        leverage_used = (
            notional_usd / float(capital_before_trade_usd)
            if capital_before_trade_usd is not None and float(capital_before_trade_usd) > 0
            else None
        )
        actual_risk_usd = (
            float(contracts) * float(risk_per_contract_usd)
            if math.isfinite(risk_per_contract_usd) and float(risk_per_contract_usd) > 0
            else None
        )
        return PositionSizingDecision(
            position_sizing_mode="fixed_contracts",
            contracts=contracts,
            contracts_raw=float(contracts),
            capital_before_trade_usd=capital_before_trade_usd,
            risk_pct=None,
            risk_budget_usd=None,
            stop_distance_points=stop_distance_points if math.isfinite(stop_distance_points) else None,
            risk_per_contract_usd=float(risk_per_contract_usd) if math.isfinite(risk_per_contract_usd) else None,
            actual_risk_usd=actual_risk_usd,
            notional_usd=notional_usd,
            leverage_used=leverage_used,
            max_contracts=None,
            skip_trade_if_too_small=None,
            skipped=False,
            skip_reason=None,
        )

    if capital_before_trade_usd is None or float(capital_before_trade_usd) <= 0:
        return PositionSizingDecision(
            position_sizing_mode="risk_percent",
            contracts=0,
            contracts_raw=None,
            capital_before_trade_usd=capital_before_trade_usd,
            risk_pct=float(config.risk_pct),
            risk_budget_usd=None,
            stop_distance_points=stop_distance_points if math.isfinite(stop_distance_points) else None,
            risk_per_contract_usd=float(risk_per_contract_usd) if math.isfinite(risk_per_contract_usd) else None,
            actual_risk_usd=None,
            notional_usd=None,
            leverage_used=None,
            max_contracts=int(config.max_contracts),
            skip_trade_if_too_small=bool(config.skip_trade_if_too_small),
            skipped=True,
            skip_reason="non_positive_capital",
        )

    risk_budget_usd = float(capital_before_trade_usd) * float(config.risk_pct)
    if not math.isfinite(risk_per_contract_usd) or float(risk_per_contract_usd) <= 0:
        return PositionSizingDecision(
            position_sizing_mode="risk_percent",
            contracts=0,
            contracts_raw=None,
            capital_before_trade_usd=float(capital_before_trade_usd),
            risk_pct=float(config.risk_pct),
            risk_budget_usd=risk_budget_usd,
            stop_distance_points=stop_distance_points if math.isfinite(stop_distance_points) else None,
            risk_per_contract_usd=float(risk_per_contract_usd) if math.isfinite(risk_per_contract_usd) else None,
            actual_risk_usd=None,
            notional_usd=None,
            leverage_used=None,
            max_contracts=int(config.max_contracts),
            skip_trade_if_too_small=bool(config.skip_trade_if_too_small),
            skipped=True,
            skip_reason="non_positive_risk_per_contract",
        )

    contracts_raw = float(risk_budget_usd) / float(risk_per_contract_usd)
    contracts = int(math.floor(contracts_raw))
    skip_reason: str | None = None

    if contracts < 1:
        if bool(config.skip_trade_if_too_small):
            skip_reason = "contracts_below_one"
            contracts = 0
        else:
            contracts = 1

    contracts = min(int(contracts), int(config.max_contracts))
    if contracts < 1:
        skip_reason = skip_reason or "contracts_below_one_after_cap"

    notional_usd = float(contracts) * float(entry_price) * float(point_value_usd) if contracts > 0 else None
    leverage_used = notional_usd / float(capital_before_trade_usd) if contracts > 0 else None
    actual_risk_usd = float(contracts) * float(risk_per_contract_usd) if contracts > 0 else None
    return PositionSizingDecision(
        position_sizing_mode="risk_percent",
        contracts=int(contracts),
        contracts_raw=contracts_raw,
        capital_before_trade_usd=float(capital_before_trade_usd),
        risk_pct=float(config.risk_pct),
        risk_budget_usd=risk_budget_usd,
        stop_distance_points=float(stop_distance_points),
        risk_per_contract_usd=float(risk_per_contract_usd),
        actual_risk_usd=actual_risk_usd,
        notional_usd=notional_usd,
        leverage_used=leverage_used,
        max_contracts=int(config.max_contracts),
        skip_trade_if_too_small=bool(config.skip_trade_if_too_small),
        skipped=bool(contracts < 1),
        skip_reason=skip_reason,
    )
