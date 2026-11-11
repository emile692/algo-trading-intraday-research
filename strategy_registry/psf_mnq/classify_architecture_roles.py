"""Classify the collaborative 134-strategy universe by architectural role.

This is a routing taxonomy, not a profitability assessment. Every strategy
receives exactly one primary role so the five buckets can be reviewed without
silently dropping ideas.
"""

from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ROLES = (
    "alpha_engine",
    "setup_trigger",
    "filter_regime",
    "level_context",
    "risk_overlay",
)


ALPHA_ENGINE = {
    # Trend engines
    "ema-ribbon-trend",
    "supertrend",
    "dual-ma-crossover",
    "triple-moving-average",
    "rainbow-ma",
    "rainbow-moving-average",
    "kama-trend",
    "kalman-filter-trend",
    "htf-trend-ltf-entry",
    # Breakout / expansion engines
    "donchian-breakout",
    "keltner-channel-breakout",
    "opening-range-breakout",
    "range-compression-breakout",
    "volatility-breakout",
    "n-bar-breakout",
    "london-breakout",
    "asian-range-breakout",
    "momentum-breakout",
    "range-midpoint-reversion",
    "consolidation-breakout",
    "compression-expansion",
    "standard-deviation-breakout",
    "atr-percentile-breakout",
    "volatility-ratio-breakout",
    # Mean-reversion engines
    "vwap-deviation",
    "rsi-bollinger-band-combo",
    "z-score-mean-reversion",
    "bollinger-band-walk",
    "regression-to-mean",
    "rsi-2-pullback",
    "z-score-reversion",
    "std-dev-bands-reversion",
    "keltner-mean-reversion",
    "envelope-channel-reversion",
    "mean-deviation-snap",
    "rsi-mean-reversion",
    # Session and short-horizon engines
    "gap-fill",
    "gap-and-go",
    "opening-gap-fade",
    "london-ny-overlap",
    "overnight-gap",
    "vwap-scalp",
    "micro-pullback-scalp",
}


FILTER_REGIME = {
    "adx-trend-strength",
    "heikin-ashi-trend",
    "hull-moving-average",
    "aroon-trend",
    "dmi-adx-system",
    "trend-intensity-index",
    "mcginley-dynamic",
    "zero-lag-ema",
    "kaufman-efficiency",
    "fractal-adaptive-ma",
    "chaikin-volatility",
    "garman-klass-volatility",
    "squeeze-momentum",
    "parkinson-volatility",
    "quad-witching",
    "mtf-macd",
    "mtf-rsi-confluence",
}


LEVEL_CONTEXT = {
    "session-high-low-breakout",
    "pivot-point-bounce",
    "darvas-box",
    "pivot-point-breakout",
    "regression-channel",
    "camarilla-breakout",
    "pivot-breakout",
    "box-range-trading",
    "vwap-anchored",
    "initial-balance-breakout",
}


RISK_OVERLAY = {
    "parabolic-sar-trend",
    "parabolic-sar-reversal",
}


SETUP_TRIGGER = {
    # Breakout timing and confirmation
    "inside-bar-breakout",
    "adl-breakout",
    "narrow-body-candle",
    # Oscillator and reversal timing
    "stochastic-rsi-reversal",
    "commodity-channel-reversal",
    "three-bar-reversal",
    "cci-divergence",
    "stochastic-pop",
    "percent-b-reversal",
    "fisher-transform",
    "detrended-oscillator",
    "percent-b-strategy",
    # Trend and momentum timing
    "macd-crossover",
    "stochastic-divergence",
    "macd-histogram-divergence",
    "dema-crossover",
    "cci-momentum",
    "rate-of-change",
    "chande-momentum",
    "tsi-crossover",
    "awesome-oscillator",
    "momentum-oscillator",
    "tsi-momentum",
    "connors-rsi",
    "ultimate-oscillator",
    "vwap-crossover",
    "vortex-crossover",
    "stochastic-momentum-index",
    "relative-vigor-index",
    "dpo-cycle",
    "rvi-divergence",
    "schaff-trend-cycle",
    "aroon-crossover",
    "rate-of-change-momentum",
    "tema-crossover",
    # Volume-derived timing and confirmation
    "obv-divergence",
    "money-flow-index",
    "klinger-oscillator",
    "volume-weighted-macd",
    "ease-of-movement",
    "balance-of-power",
    "vw-macd",
    "volume-climax-reversal",
    "force-index-breakout",
    # Price-action and exhaustion patterns
    "bullish-engulfing",
    "hammer-reversal",
    "marubozu-momentum",
    "spinning-top-reversal",
    "hidden-bullish-divergence",
    "hidden-bearish-divergence",
    "volume-price-divergence",
    # Chart patterns
    "gartley-pattern",
    "butterfly-pattern",
    "bat-pattern",
    "shark-pattern",
    "abcd-pattern",
    "cup-and-handle",
    "double-top-bottom",
    "ascending-triangle",
    "descending-triangle",
    "bull-flag-pattern",
    "wedge-pattern",
}


ROLE_MEMBERS = {
    "alpha_engine": ALPHA_ENGINE,
    "setup_trigger": SETUP_TRIGGER,
    "filter_regime": FILTER_REGIME,
    "level_context": LEVEL_CONTEXT,
    "risk_overlay": RISK_OVERLAY,
}

ROLE_RATIONALE = {
    "alpha_engine": (
        "Hypothèse directionnelle autonome; à tester comme source principale du PnL."
    ),
    "setup_trigger": (
        "Événement local d'entrée ou de confirmation; nécessite un moteur et un contexte explicites."
    ),
    "filter_regime": (
        "Décrit surtout l'état, la direction ou la volatilité du marché; à tester comme condition d'activation."
    ),
    "level_context": (
        "La contribution distinctive est le calcul d'un niveau ou d'une zone de référence."
    ),
    "risk_overlay": (
        "La contribution distinctive est une règle dynamique de sortie ou de stop après l'entrée."
    ),
}


def validate(universe: set[str]) -> None:
    assignments: dict[str, list[str]] = {}
    for role, slugs in ROLE_MEMBERS.items():
        for slug in slugs:
            assignments.setdefault(slug, []).append(role)

    duplicates = {slug: roles for slug, roles in assignments.items() if len(roles) != 1}
    if duplicates:
        raise ValueError(f"Multiple primary roles: {duplicates}")

    classified = set(assignments)
    missing = universe - classified
    unknown = classified - universe
    if missing or unknown:
        raise ValueError(
            f"Role map does not match collaborative universe; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def main() -> None:
    source = ROOT / "collaborative_universe.csv"
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    universe = {row["slug"] for row in rows}
    if len(rows) != 134 or len(universe) != 134:
        raise ValueError("Expected the jointly filtered 134-strategy universe")
    validate(universe)

    role_by_slug = {
        slug: role for role, slugs in ROLE_MEMBERS.items() for slug in slugs
    }
    output_rows = []
    for row in rows:
        role = role_by_slug[row["slug"]]
        output_rows.append(
            {
                "slug": row["slug"],
                "name": row["name"],
                "architecture_role": role,
                "research_family": row["research_family"],
                "classification_status": "draft_to_review",
                "rationale": ROLE_RATIONALE[role],
                "card_path": row["card_path"],
            }
        )

    output = ROOT / "architecture_roles.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    counts = Counter(row["architecture_role"] for row in output_rows)
    print("Architecture roles:", {role: counts[role] for role in ROLES})
    print("Total:", sum(counts.values()))


if __name__ == "__main__":
    main()
