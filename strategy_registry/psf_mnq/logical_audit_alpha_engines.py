"""Run the logic-only audit of the 43 provisional alpha engines.

The audit deliberately ignores every published and local performance metric.
It evaluates autonomy, hypothesis clarity, mechanism uniqueness, complexity,
and MNQ intraday compatibility from the published rule cards only.
"""

from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def decision(
    status: str,
    cluster: str,
    representative: str,
    autonomy: str,
    hypothesis: str,
    complexity: str,
    intraday: str,
    reason: str,
    reviewed_family: str = "",
) -> dict[str, str]:
    return {
        "logical_status": status,
        "mechanism_cluster": cluster,
        "representative_slug": representative,
        "autonomy_test": autonomy,
        "hypothesis_test": hypothesis,
        "complexity_grade": complexity,
        "intraday_test": intraday,
        "review_reason": reason,
        "reviewed_family": reviewed_family,
    }


# One explicit decision per provisional alpha engine. A representative is not a
# profitability winner: it is the canonical rule chosen to count one hypothesis.
DECISIONS = {
    "ema-ribbon-trend": decision(
        "alpha_representative", "trend_pullback_ma_structure", "ema-ribbon-trend",
        "pass", "pass", "C", "conditional_session_flatten",
        "Structure de tendance et pullback produisent direction, entrée, sortie et invalidation.",
    ),
    "vwap-deviation": decision(
        "alpha_representative", "vwap_equilibrium_reversion", "vwap-deviation",
        "pass", "pass", "B", "conditional_session_flatten",
        "Écart à l'équilibre intraday, cible VWAP et invalidation à la bande externe forment une hypothèse complète.",
    ),
    "donchian-breakout": decision(
        "alpha_representative", "rolling_range_breakout", "donchian-breakout",
        "pass", "pass", "B", "conditional_session_flatten",
        "Représentant canonique de la continuation après sortie d'un range roulant.",
    ),
    "keltner-channel-breakout": decision(
        "alpha_variant", "rolling_range_breakout", "donchian-breakout",
        "pass", "pass", "B", "conditional_session_flatten",
        "Même hypothèse de sortie de range que Donchian, avec une enveloppe définie par EMA et ATR.",
    ),
    "supertrend": decision(
        "alpha_representative", "atr_trend_flip", "supertrend",
        "pass", "pass", "A", "conditional_session_flatten",
        "Le basculement d'un état de tendance ATR fournit directement position et sortie opposée.",
    ),
    "dual-ma-crossover": decision(
        "alpha_representative", "ma_crossover_trend", "dual-ma-crossover",
        "pass", "pass", "B", "conditional_session_flatten",
        "Représentant simple de la persistance de tendance mesurée par croisement de moyennes.",
    ),
    "opening-range-breakout": decision(
        "alpha_representative", "session_range_breakout", "opening-range-breakout",
        "pass", "pass", "B", "conditional_session_flatten",
        "L'ouverture définit un range causal puis son franchissement porte l'hypothèse de continuation intraday.",
    ),
    "rsi-bollinger-band-combo": decision(
        "alpha_variant", "channel_extreme_reversion", "keltner-mean-reversion",
        "pass", "pass", "C", "conditional_session_flatten",
        "Réversion d'un extrême de canal déjà représentée; RSI et confirmation ajoutent des conditions imbriquées.",
    ),
    "range-compression-breakout": decision(
        "alpha_representative", "compression_breakout", "range-compression-breakout",
        "pass", "pass", "C", "conditional_session_flatten",
        "Représentant de l'hypothèse compression mesurable puis expansion directionnelle.",
    ),
    "triple-moving-average": decision(
        "alpha_variant", "trend_pullback_ma_structure", "ema-ribbon-trend",
        "pass", "pass", "C", "conditional_session_flatten",
        "Même tendance ordonnée et même entrée sur pullback que l'EMA Ribbon, avec trois moyennes.",
    ),
    "z-score-mean-reversion": decision(
        "alpha_representative", "statistical_zscore_reversion", "z-score-mean-reversion",
        "pass", "pass", "B", "conditional_session_flatten",
        "Représentant falsifiable de la réversion d'un écart normalisé à une moyenne roulante.",
    ),
    "bollinger-band-walk": decision(
        "demote_trigger", "exhaustion_reversal_trigger", "",
        "partial", "pass", "C", "conditional_formalization",
        "La marche de bande, la divergence RSI et la baisse de volume décrivent un setup d'épuisement; la divergence reste à formaliser.",
        "reversal_exhaustion",
    ),
    "regression-to-mean": decision(
        "alpha_representative", "regression_equilibrium_reversion", "regression-to-mean",
        "pass", "pass", "C", "conditional_session_flatten",
        "L'équilibre estimé par régression et ses bandes définissent une hypothèse distincte du simple z-score roulant.",
    ),
    "volatility-breakout": decision(
        "alpha_representative", "volatility_breakout", "volatility-breakout",
        "pass", "pass", "B", "pass",
        "Le déplacement depuis l'ouverture, normalisé par l'ATR précédent, définit entrée, direction et invalidation intraday.",
    ),
    "n-bar-breakout": decision(
        "alpha_variant", "rolling_range_breakout", "donchian-breakout",
        "pass", "pass", "B", "conditional_session_flatten",
        "Formulation générique du Donchian breakout; le mécanisme économique est identique.",
    ),
    "london-breakout": decision(
        "alpha_variant", "session_range_breakout", "opening-range-breakout",
        "pass", "pass", "B", "pass",
        "Variante de breakout d'un range de session, ancrée sur la transition Asie-Londres.",
    ),
    "asian-range-breakout": decision(
        "alpha_variant", "session_range_breakout", "opening-range-breakout",
        "pass", "pass", "C", "pass",
        "Même mécanisme de range de session que l'ORB, avec fenêtres et filtres supplémentaires.",
    ),
    "standard-deviation-breakout": decision(
        "alpha_variant", "volatility_breakout", "volatility-breakout",
        "pass", "pass", "C", "conditional_session_flatten",
        "Déclenche une continuation sur mouvement anormal; la normalisation change mais pas l'hypothèse de choc de volatilité.",
        "breakout_expansion",
    ),
    "rsi-2-pullback": decision(
        "alpha_representative", "short_horizon_pullback_reversion", "rsi-2-pullback",
        "pass", "pass", "C", "conditional_session_flatten",
        "Représentant d'une réversion courte dans le sens d'un filtre de tendance long terme.",
    ),
    "z-score-reversion": decision(
        "merge_duplicate", "statistical_zscore_reversion", "z-score-mean-reversion",
        "pass", "pass", "B", "conditional_session_flatten",
        "Même nom publié et mêmes règles de z-score, seuils d'entrée, cible et invalidation.",
    ),
    "atr-percentile-breakout": decision(
        "alpha_variant", "compression_breakout", "range-compression-breakout",
        "pass", "pass", "C", "conditional_session_flatten",
        "La compression est mesurée par percentile d'ATR avant un breakout court; variante du même mécanisme.",
        "breakout_expansion",
    ),
    "rainbow-ma": decision(
        "alpha_variant", "trend_pullback_ma_structure", "ema-ribbon-trend",
        "pass", "pass", "C", "conditional_session_flatten",
        "Même structure de moyennes ordonnées et pullback que l'EMA Ribbon, avec davantage de couches.",
    ),
    "rainbow-moving-average": decision(
        "merge_duplicate", "trend_pullback_ma_structure", "rainbow-ma",
        "pass", "pass", "C", "conditional_session_flatten",
        "Même nom publié, même empilement de dix moyennes et même entrée sur pullback que Rainbow MA.",
    ),
    "std-dev-bands-reversion": decision(
        "alpha_variant", "statistical_zscore_reversion", "z-score-mean-reversion",
        "pass", "pass", "C", "conditional_session_flatten",
        "Les bandes sigma expriment le même écart normalisé; paliers et scaling augmentent seulement la complexité.",
    ),
    "momentum-breakout": decision(
        "alpha_representative", "momentum_confirmed_breakout", "momentum-breakout",
        "pass", "pass", "C", "conditional_session_flatten",
        "Représentant du breakout conditionné par force directionnelle et momentum, à compter séparément du breakout nu.",
    ),
    "volatility-ratio-breakout": decision(
        "alpha_variant", "volatility_breakout", "volatility-breakout",
        "pass", "pass", "C", "conditional_session_flatten",
        "Le ratio de range détecte le même choc directionnel que le moteur de volatility breakout.",
        "breakout_expansion",
    ),
    "vwap-scalp": decision(
        "demote_trigger", "vwap_equilibrium_reversion", "vwap-deviation",
        "partial", "pass", "B", "conditional_remove_non_ohlcv",
        "La logique réutilisable est le timing de retournement à une bande VWAP; Tick Chart et Volume Delta ne doivent pas être reconstruits.",
    ),
    "micro-pullback-scalp": decision(
        "demote_trigger", "trend_pullback_ma_structure", "ema-ribbon-trend",
        "partial", "pass", "C", "conditional_formalization",
        "Le pullback sur EMA est un déclencheur; pente 'forte', micro-swing et objectif variable empêchent d'en faire un moteur autonome figé.",
    ),
    "htf-trend-ltf-entry": decision(
        "demote_filter", "multi_timeframe_trend_filter", "",
        "fail", "pass", "C", "fail_undefined_levels",
        "Le biais Daily par EMA est un filtre exploitable, mais support, résistance et pullback LTF ne sont pas définis mécaniquement.",
    ),
    "gap-fill": decision(
        "alpha_representative", "gap_fade", "gap-fill",
        "pass", "pass", "B", "pass",
        "Petit gap, direction opposée, cible prior close, invalidation et time stop forment un moteur autonome.",
    ),
    "gap-and-go": decision(
        "alpha_representative", "gap_continuation", "gap-and-go",
        "pass", "pass", "C", "conditional_session_flatten",
        "Grand gap confirmé par la première barre et le volume porte une hypothèse de continuation distincte du gap fade.",
    ),
    "opening-gap-fade": decision(
        "alpha_variant", "gap_fade", "gap-fill",
        "pass", "pass", "B", "pass",
        "Même hypothèse de comblement du gap, avec la première barre de cinq minutes comme confirmation.",
    ),
    "kama-trend": decision(
        "alpha_variant", "ma_crossover_trend", "dual-ma-crossover",
        "pass", "pass", "B", "conditional_session_flatten",
        "Même persistance de tendance et même logique de franchissement, avec moyenne adaptative.",
    ),
    "range-midpoint-reversion": decision(
        "alpha_representative", "range_structure_reversion", "range-midpoint-reversion",
        "pass", "pass", "B", "conditional_session_flatten",
        "Achète et vend les extrêmes d'un range causal avec son milieu pour cible; ce n'est pas un breakout.",
        "mean_reversion",
    ),
    "consolidation-breakout": decision(
        "alpha_variant", "compression_breakout", "range-compression-breakout",
        "pass", "pass", "C", "conditional_session_flatten",
        "Contraction prolongée puis clôture hors range: variante moins précisément définie du moteur de compression.",
    ),
    "compression-expansion": decision(
        "alpha_variant", "compression_breakout", "range-compression-breakout",
        "pass", "pass", "C", "conditional_session_flatten",
        "ATR contracté puis expansion avec biais EMA exprime le même mécanisme avec plus de degrés de liberté.",
    ),
    "keltner-mean-reversion": decision(
        "alpha_representative", "channel_extreme_reversion", "keltner-mean-reversion",
        "pass", "pass", "C", "conditional_session_flatten",
        "Représentant de la réversion d'un extrême de canal volatil vers sa ligne centrale.",
    ),
    "envelope-channel-reversion": decision(
        "alpha_variant", "channel_extreme_reversion", "keltner-mean-reversion",
        "pass", "pass", "B", "conditional_session_flatten",
        "Même réversion d'extrême de canal, avec largeur fixe en pourcentage au lieu d'une largeur ATR.",
    ),
    "mean-deviation-snap": decision(
        "alpha_variant", "statistical_zscore_reversion", "z-score-mean-reversion",
        "pass", "pass", "C", "conditional_session_flatten",
        "Règles presque identiques au z-score représentatif; le scaling à 2,5 sigma est la différence principale.",
    ),
    "kalman-filter-trend": decision(
        "reject_logic", "ma_crossover_trend", "dual-ma-crossover",
        "partial", "pass", "C", "fail_unspecified_model",
        "Le modèle d'état, les bruits du filtre et le seuil de pente 'près de zéro' manquent; les choisir inventerait la stratégie.",
        "trend_momentum",
    ),
    "london-ny-overlap": decision(
        "alpha_variant", "session_range_breakout", "opening-range-breakout",
        "pass", "pass", "C", "pass",
        "Breakout d'un opening range sur une autre fenêtre de liquidité; les filtres supplémentaires n'en font pas une hypothèse indépendante.",
    ),
    "overnight-gap": decision(
        "demote_filter", "gap_regime_router", "",
        "pass", "fail_multiple_hypotheses", "C", "conditional_session_flatten",
        "La fiche route petit gap vers Gap Fill et grand gap vers Gap & Go; elle combine deux moteurs déjà représentés.",
    ),
    "rsi-mean-reversion": decision(
        "alpha_variant", "short_horizon_pullback_reversion", "rsi-2-pullback",
        "pass", "pass", "C", "conditional_session_flatten",
        "Même pullback contrarien filtré par tendance que RSI(2), avec oscillateur plus lent.",
    ),
}


RESULTING_ROLE = {
    "alpha_representative": "alpha_engine",
    "alpha_variant": "alpha_engine",
    "demote_trigger": "setup_trigger",
    "demote_filter": "filter_regime",
    "merge_duplicate": "merged_duplicate",
    "defer_horizon": "alpha_engine_deferred",
    "reject_logic": "rejected",
}

UNIQUENESS = {
    "alpha_representative": "representative",
    "alpha_variant": "variant",
    "demote_trigger": "component",
    "demote_filter": "component",
    "merge_duplicate": "duplicate",
    "defer_horizon": "unresolved",
    "reject_logic": "unresolved",
}


def main() -> None:
    source = ROOT / "architecture_roles.csv"
    with source.open(encoding="utf-8", newline="") as handle:
        alpha_rows = [
            row for row in csv.DictReader(handle)
            if row["architecture_role"] == "alpha_engine"
        ]

    slugs = {row["slug"] for row in alpha_rows}
    if len(alpha_rows) != 43 or len(slugs) != 43:
        raise ValueError("Expected 43 provisional alpha engines")
    if slugs != set(DECISIONS):
        raise ValueError(
            f"Audit map mismatch; missing={sorted(slugs - set(DECISIONS))}, "
            f"unknown={sorted(set(DECISIONS) - slugs)}"
        )

    output_rows = []
    for row in alpha_rows:
        audit = DECISIONS[row["slug"]]
        status = audit["logical_status"]
        reviewed_family = audit["reviewed_family"] or row["research_family"]
        output_rows.append(
            {
                "slug": row["slug"],
                "name": row["name"],
                "source_family": row["research_family"],
                "reviewed_family": reviewed_family,
                "logical_status": status,
                "resulting_role": RESULTING_ROLE[status],
                "mechanism_cluster": audit["mechanism_cluster"],
                "representative_slug": audit["representative_slug"],
                "autonomy_test": audit["autonomy_test"],
                "hypothesis_test": audit["hypothesis_test"],
                "uniqueness_test": UNIQUENESS[status],
                "complexity_grade": audit["complexity_grade"],
                "intraday_test": audit["intraday_test"],
                "review_reason": audit["review_reason"],
                "card_path": row["card_path"],
            }
        )

    output = ROOT / "alpha_engine_logical_audit.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    statuses = Counter(row["logical_status"] for row in output_rows)
    complexities = Counter(row["complexity_grade"] for row in output_rows)
    retained = [
        row for row in output_rows
        if row["logical_status"] in {"alpha_representative", "alpha_variant"}
    ]
    representatives = [
        row for row in output_rows
        if row["logical_status"] == "alpha_representative"
    ]
    if len({row["mechanism_cluster"] for row in representatives}) != len(representatives):
        raise ValueError("Each retained representative must own one distinct mechanism cluster")

    representative_by_cluster = {
        row["mechanism_cluster"]: row for row in representatives
    }
    mechanism_rows = []
    for representative in representatives:
        cluster = representative["mechanism_cluster"]
        members = [
            row for row in retained
            if row["mechanism_cluster"] == cluster
        ]
        variants = [
            row for row in members
            if row["logical_status"] == "alpha_variant"
        ]
        mechanism_rows.append(
            {
                "mechanism_cluster": cluster,
                "representative_slug": representative["slug"],
                "representative_name": representative["name"],
                "reviewed_family": representative["reviewed_family"],
                "alpha_engine_count": len(members),
                "variant_count": len(variants),
                "variant_slugs": "|".join(row["slug"] for row in variants),
                "complexity_grades": "|".join(
                    sorted({row["complexity_grade"] for row in members})
                ),
            }
        )

    retained_clusters = {row["mechanism_cluster"] for row in retained}
    if retained_clusters != set(representative_by_cluster):
        raise ValueError("Every retained alpha cluster must have one representative")

    mechanism_output = ROOT / "alpha_engine_mechanisms.csv"
    with mechanism_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mechanism_rows[0]))
        writer.writeheader()
        writer.writerows(mechanism_rows)

    print("Logical statuses:", dict(sorted(statuses.items())))
    print("Complexity grades:", dict(sorted(complexities.items())))
    print("Retained unique alpha engines:", len(retained))
    print("Distinct represented mechanisms:", len(representatives))


if __name__ == "__main__":
    main()
