"""Create a reproducible first-pass research triage from the dated card index.

The decisions are research routing only. They are not claims of profitability.
"""

from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent

# A panel of distinct, initially testable mechanisms. The order is a work queue,
# not a performance ranking. Features are assessed as regime descriptors.
SHORTLIST = [
    ("gap-fill", "prototype", "Session gap fade; first standalone baseline and potential complement to ORB."),
    ("vwap-deviation", "prototype", "Intraday reversion around an observable price-volume reference."),
    ("liquidity-sweep-reversal", "prototype", "Failed-breakout reversal, structurally different from ORB continuation."),
    ("gap-and-go", "prototype", "Gap continuation; quantify overlap with ORB rather than assume diversification."),
    ("three-bar-reversal", "prototype", "Simple candle reversal with few required data fields."),
    ("keltner-channel-breakout", "prototype", "Channel breakout representative of trend/expansion ideas."),
    ("volatility-breakout", "prototype", "Volatility trigger on intraday bars, distinct from a fixed opening range."),
    ("range-midpoint-reversion", "prototype", "Structural range reversion, subject to a precise range definition."),
    ("pivot-point-bounce", "prototype", "Level-based reversion with prior-session information."),
    ("rsi-2-pullback", "prototype", "Short-horizon oscillator reversion representative."),
    ("volume-climax-reversal", "prototype", "Price-volume exhaustion using available OHLCV bars."),
    ("narrow-body-candle", "prototype", "Simple compression proxy with an intraday cited horizon."),
    ("camarilla-breakout", "prototype", "Prior-session pivot breakout, separate from an opening-range breakout."),
    ("parkinson-volatility", "feature", "High-low volatility feature; test as a regime variable, not standalone alpha."),
    ("kaufman-efficiency", "feature", "Trendiness feature; evaluate only with information available at signal time."),
]
SHORTLIST_INDEX = {slug: (rank, kind, reason) for rank, (slug, kind, reason) in enumerate(SHORTLIST, 1)}

POSSIBLE_DUPLICATE_OF = {
    "z-score-reversion": "z-score-mean-reversion",
    "pivot-breakout": "pivot-point-breakout",
    "rainbow-moving-average": "rainbow-ma",
    "rvi-divergence": "relative-vigor-index",
}

# Clusters describe broad mechanisms. They do not assert identical rules.
CLUSTER_OVERRIDES = {
    "gap_fade": {"gap-fill", "opening-gap-fade", "overnight-gap"},
    "gap_continuation": {"gap-and-go"},
    "opening_range_breakout": {"opening-range-breakout", "initial-balance-breakout", "session-high-low-breakout"},
    "compression_expansion": {"range-compression-breakout", "inside-bar-breakout", "narrow-body-candle", "compression-expansion", "squeeze-momentum", "consolidation-breakout"},
    "volatility_channel_breakout": {"keltner-channel-breakout", "donchian-breakout", "standard-deviation-breakout", "volatility-breakout", "volatility-ratio-breakout", "atr-percentile-breakout"},
    "pivot_levels": {"pivot-point-bounce", "pivot-point-breakout", "pivot-breakout", "camarilla-breakout"},
    "vwap_reversion": {"vwap-deviation", "vwap-scalp"},
    "statistical_reversion": {"z-score-mean-reversion", "z-score-reversion", "regression-to-mean", "mean-deviation-snap", "rsi-mean-reversion", "rsi-2-pullback"},
    "band_reversion": {"keltner-mean-reversion", "envelope-channel-reversion", "std-dev-bands-reversion", "percent-b-reversal", "percent-b-strategy"},
    "failed_breakout_reversal": {"liquidity-sweep-reversal", "wyckoff-spring", "wyckoff-upthrust", "ict-judas-swing"},
    "volatility_measure": {"parkinson-volatility", "garman-klass-volatility", "chaikin-volatility"},
    "trend_efficiency": {"kaufman-efficiency", "kama-trend", "fractal-adaptive-ma", "kalman-filter-trend"},
    "market_profile": {"volume-profile-poc", "value-area-fade", "poc-migration", "poor-high-low", "single-print-defense"},
    "money_management": {"grid-trading", "martingale-strategy", "anti-martingale"},
}

ORDER_FLOW_DATA = {
    "bid-ask-imbalance", "iceberg-order-detection", "spoofing-detection",
    "tick-imbalance-bars", "market-impact-reversal", "cumulative-delta-divergence",
    "volume-imbalance-stacking", "absorption-pattern", "aggressive-exhaustion",
    "delta-flip-momentum", "sweeping-the-book", "order-flow-delta-divergence",
    "tape-reading-scalp", "order-flow-scalp", "spread-scalp", "tick-chart-momentum",
}
PROFILE_DATA = CLUSTER_OVERRIDES["market_profile"]
EXTERNAL_EVENT_DATA = {"nfp-fade"}
SUBJECTIVE_PATTERN_FAMILIES = {"price_action_reversal", "chart_pattern"}
USER_EXCLUDED_ICT_ADJACENT = {
    "market-structure-shift",
    "liquidity-sweep-reversal",
    "supply-demand-zones",
    "wyckoff-accumulation",
    "wyckoff-spring",
    "wyckoff-upthrust",
}
MISSING_NON_OHLCV_DATA = {
    "tick_or_order_book_missing",
    "volume_at_price_missing",
    "historical_event_calendar_missing",
}


def cluster(row: dict[str, str]) -> str:
    slug = row["slug"]
    matches = [key for key, members in CLUSTER_OVERRIDES.items() if slug in members]
    if len(matches) > 1:
        raise ValueError(f"Overlapping cluster overrides for {slug}: {matches}")
    return matches[0] if matches else row["research_family"]


def data_gate(row: dict[str, str]) -> str:
    slug = row["slug"]
    if slug in ORDER_FLOW_DATA:
        return "tick_or_order_book_missing"
    if slug in PROFILE_DATA:
        return "volume_at_price_missing"
    if slug in EXTERNAL_EVENT_DATA:
        return "historical_event_calendar_missing"
    horizons = row["timeframes"].split(",")
    if not set(horizons) & {"1min", "5min", "15min", "30min"}:
        return "intraday_horizon_review"
    return "ohlcv_available"


def triage(row: dict[str, str]) -> dict[str, str | int]:
    slug = row["slug"]
    gate = data_gate(row)
    rank, kind, shortlist_reason = SHORTLIST_INDEX.get(slug, ("", "", ""))
    if rank:
        if slug == "gap-fill":
            stage = "is_screen_failed"
            reason = "Prespecified v0 has negative net expectancy on IS; see experiments/gap_fill_v0/README.md."
        else:
            stage = "feature_queue" if kind == "feature" else "prototype_queue"
            reason = shortlist_reason
        if gate not in {"ohlcv_available", "intraday_horizon_review"}:
            raise ValueError(f"Shortlisted strategy has unavailable data: {slug}: {gate}")
    elif row["research_role"] == "exclude_as_alpha":
        stage, reason = "exclude_as_alpha", "Sizing or grid scheme, not a source of entry alpha."
    elif slug == "opening-range-breakout":
        stage, reason = "existing_reference", "Site ORB is close to the retained local ORB; no new independent sleeve."
    elif gate.endswith("missing"):
        stage, reason = "defer_data", f"Required dataset is unavailable: {gate}."
    elif slug in POSSIBLE_DUPLICATE_OF:
        stage, reason = "review_duplicate", f"Same published name as {POSSIBLE_DUPLICATE_OF[slug]}; compare rules before merging."
    elif gate == "intraday_horizon_review":
        stage, reason = "defer_horizon", "Published horizons start at 1H or slower; intraday prop-firm fit needs review."
    elif row["research_family"] in SUBJECTIVE_PATTERN_FAMILIES:
        stage, reason = "defer_formalization", "Pattern boundaries and timing need an unambiguous, causal definition."
    elif row["research_role"] in {"feature_or_filter", "alpha_or_filter", "meta_signal"}:
        stage, reason = "feature_pool", "Consider as a regime feature or filter after representative alpha tests."
    else:
        stage, reason = "defer_overlap", "Keep as a related idea; compare its rules with the selected mechanism representative."
    return {
        "slug": slug,
        "name": row["name"],
        "research_family": row["research_family"],
        "cluster_preliminary": cluster(row),
        "data_gate": gate,
        "stage": stage,
        "queue_order": rank,
        "possible_duplicate_of": POSSIBLE_DUPLICATE_OF.get(slug, ""),
        "reason": reason,
        "card_path": row["card_path"],
    }


def main() -> None:
    with (ROOT / "index.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 185 or len({row["slug"] for row in rows}) != 185:
        raise ValueError("Expected the dated 185-card registry")
    if len(SHORTLIST_INDEX) != len(SHORTLIST):
        raise ValueError("Shortlist contains duplicate slugs")
    missing = set(SHORTLIST_INDEX) - {row["slug"] for row in rows}
    if missing:
        raise ValueError(f"Shortlist entries missing from registry: {sorted(missing)}")
    result = [triage(row) for row in rows]
    output = ROOT / "triage.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    eligible = [
        row for row in result
        if row["data_gate"] not in MISSING_NON_OHLCV_DATA
        and not row["slug"].startswith("ict-")
        and row["slug"] not in USER_EXCLUDED_ICT_ADJACENT
        and row["research_family"] != "money_management"
    ]
    with (ROOT / "collaborative_universe.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(eligible)
    stages = Counter(row["stage"] for row in result)
    gates = Counter(row["data_gate"] for row in result)
    print("Triage stages:", dict(sorted(stages.items())))
    print("Data gates:", dict(sorted(gates.items())))
    print("Shortlist:", len([row for row in result if row["queue_order"]]))
    print("Collaborative OHLCV, no-ICT universe:", len(eligible))


if __name__ == "__main__":
    main()
