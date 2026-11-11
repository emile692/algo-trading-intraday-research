"""Build the dated PineScriptForge MNQ research cards from a downloaded site bundle.

Usage: python build_cards.py path/to/index-*.js

The bundle is an input only: it is not committed to the research repository.
Requires the small ``json5`` package to parse the site's JavaScript literals.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import json5


ROOT = Path(__file__).resolve().parent
ASSET_URL = "https://pinescriptforge.com/assets/index-DJyf-tLw.js"
SITE = "https://pinescriptforge.com"
SNAPSHOT_DATE = "2026-09-18"
SUPPLEMENTAL_SLUG = "rsi-mean-reversion"

# These are the first-timeframe ranges in the site bundle's displayed-results
# generator. The output below is used only to reproduce catalogue membership.
# Its synthetic trade counts and all resulting performance figures are discarded.
TRADES_PER_MONTH_RANGES = {
    "1min": (15, 25),
    "5min": (12, 22),
    "15min": (8, 18),
    "1H": (5, 12),
    "4H": (3, 8),
    "Daily": (1, 4),
    "Weekly": (0.5, 2),
}

FAMILY = {
    "time-based": ("session_gap", "alpha_candidate"),
    "gap": ("session_gap", "alpha_candidate"),
    "breakout": ("breakout_expansion", "alpha_candidate"),
    "range": ("breakout_expansion", "alpha_candidate"),
    "mean-reversion": ("mean_reversion", "alpha_candidate"),
    "trend": ("trend_momentum", "alpha_or_filter"),
    "momentum": ("trend_momentum", "alpha_or_filter"),
    "divergence": ("reversal_exhaustion", "alpha_candidate"),
    "smart-money": ("price_action_reversal", "alpha_candidate"),
    "wyckoff": ("price_action_reversal", "alpha_candidate"),
    "price-action": ("price_action", "alpha_candidate"),
    "pattern-recognition": ("chart_pattern", "alpha_candidate"),
    "harmonic": ("chart_pattern", "alpha_candidate"),
    "elliott-wave": ("chart_pattern", "alpha_candidate"),
    "volatility": ("volatility_regime", "feature_or_filter"),
    "vol-regime": ("volatility_regime", "feature_or_filter"),
    "options-vol": ("volatility_regime", "feature_or_filter"),
    "volume": ("volume_auction", "alpha_or_feature"),
    "auction-market": ("volume_auction", "alpha_or_feature"),
    "microstructure": ("order_flow", "data_gated_candidate"),
    "advanced-order-flow": ("order_flow", "data_gated_candidate"),
    "statistical": ("statistical", "alpha_or_feature"),
    "intermarket": ("cross_asset", "alpha_candidate"),
    "seasonal": ("calendar", "alpha_candidate"),
    "sentiment": ("sentiment", "feature_or_filter"),
    "multi-timeframe": ("composite", "meta_signal"),
    "scalping": ("short_horizon", "alpha_candidate"),
}


def source_strategies(bundle: str) -> list[dict]:
    """Read the site's static 310-strategy descriptions, not its scores."""
    start = bundle.index("il=[{") + len("il=")
    end = bundle.index("],gy=", start) + 1
    strategies = json5.loads(bundle[start:end])
    if not isinstance(strategies, list) or len(strategies) < 185:
        raise ValueError("Unexpected strategy array in site bundle")
    slugs = [item["slug"] for item in strategies]
    if len(slugs) != len(set(slugs)):
        raise ValueError("Duplicate source slugs")
    return strategies


def displayed_trade_count(slug: str, first_timeframe: str) -> int:
    """Reproduce only the site's 300-trade inclusion filter.

    This count comes from a deterministic simulator, not a market backtest.
    Do not write it to a card or use it to prioritize a strategy.
    """
    hash_value = 0
    for char in f"{slug}-MNQ":
        hash_value = ((hash_value << 5) - hash_value + ord(char)) & 0xFFFFFFFF
    if hash_value >= 2**31:
        hash_value -= 2**32
    value = math.sin(hash_value + 5) * 10_000
    fraction = value - math.floor(value)
    low, high = TRADES_PER_MONTH_RANGES.get(first_timeframe, (10, 35))
    monthly_trades = math.floor(low + fraction * (high - low))
    return 38 * monthly_trades


def research_classification(item: dict) -> tuple[str, str]:
    slug = item["slug"]
    if any(word in slug for word in ("martingale", "grid-trading")):
        return "money_management", "exclude_as_alpha"
    return FAMILY[item["category"]]


def data_notes(item: dict) -> list[str]:
    category = item["category"]
    words = " ".join(
        [item["name"], *item["indicators"], item["entryLogic"], item["exitLogic"]]
    ).lower()
    notes = ["Barres OHLCV adaptées aux horizons indiqués, à confirmer."]
    if category in {"time-based", "gap", "seasonal"} or any(
        term in words for term in ("session", "overnight", "opening gap")
    ):
        notes.append("Calendrier de séance et horodatage New York à définir.")
    if category in {"microstructure", "advanced-order-flow"} or any(
        term in words for term in ("order book", "level 2", "iceberg", "spoofing", "bid-ask")
    ):
        notes.append("Vérifier la nécessité de données tick, carnet ou ordres; pas de substitution OHLCV implicite.")
    if category in {"intermarket", "options-vol"}:
        notes.append("Séries externes synchronisées à identifier avant tout test.")
    if category == "sentiment":
        notes.append("Historique de sentiment ou d'événements daté à identifier.")
    if "profile" in words or "poc" in words or "value area" in words:
        notes.append("Distribution volume-prix nécessaire pour reproduire un vrai profil de marché.")
    return notes


def render_card(item: dict, membership: str, duplicate_names: dict[str, list[str]]) -> str:
    slug, name = item["slug"], item["name"]
    family, role = research_classification(item)
    strategy_url = f"{SITE}/strategy/{slug}"
    mnq_url = f"{SITE}/mnq/{slug}/backtest"
    duplicate = duplicate_names.get(name, [])
    lines = [
        f"# {name}",
        "",
        f"- **ID** : `{slug}`",
        f"- **Univers** : `{membership}`",
        "- **Statut** : `Idea` — aucune validation indépendante",
        f"- **Catégorie du site** : `{item['category']}`",
        f"- **Famille de recherche (provisoire)** : `{family}`",
        f"- **Rôle envisagé (provisoire)** : `{role}`",
        f"- **Sources** : [fiche stratégie]({strategy_url}) · [page MNQ]({mnq_url})",
        "",
        "## Description publiée",
        "",
        item["shortDescription"],
        "",
        f"- **Entrée annoncée** : {item['entryLogic']}",
        f"- **Sortie annoncée** : {item['exitLogic']}",
        "",
        "## Données et formalisation",
        "",
        f"- **Indicateurs cités** : {', '.join(item['indicators']) or 'aucun'}",
        f"- **Horizons cités** : {', '.join(item['timeframes']) or 'aucun'}",
        *[f"- **Donnée à vérifier** : {note}" for note in data_notes(item)],
        "- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).",
    ]
    if duplicate:
        lines.append(f"- **Nom partagé avec** : {', '.join(f'`{other}`' for other in duplicate if other != slug)}. Dédupliquer la logique avant expérimentation.")
    if slug == "opening-range-breakout":
        lines.append("- **Référence locale** : cette fiche du site est une idée externe; elle ne désigne pas l'ORB MNQ retenu dans ce dépôt.")
    if membership == "supplemental_page":
        lines.append("- **Hors filtre du classement courant** : page MNQ publiée, ajoutée pour compléter les 185 fiches demandées.")
    lines += [
        "",
        "Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.",
        "",
    ]
    return "\n".join(lines)


def build(bundle_path: Path) -> None:
    raw = bundle_path.read_bytes()
    all_items = source_strategies(raw.decode("utf-8"))
    ranked = [
        item for item in all_items
        if displayed_trade_count(item["slug"], item["timeframes"][0]) >= 300
    ]
    if len(ranked) != 184:
        raise ValueError(f"Expected 184 current ranked entries, found {len(ranked)}")
    supplementary = next(item for item in all_items if item["slug"] == SUPPLEMENTAL_SLUG)
    if supplementary in ranked:
        raise ValueError("Supplemental strategy is now in the ranking; reassess the snapshot")
    selected = [(item, "ranked_184") for item in ranked] + [(supplementary, "supplemental_page")]

    names: dict[str, list[str]] = {}
    for item, _ in selected:
        names.setdefault(item["name"], []).append(item["slug"])
    duplicates = {name: slugs for name, slugs in names.items() if len(slugs) > 1}

    cards = ROOT / "cards"
    cards.mkdir(exist_ok=True)
    for item, membership in selected:
        (cards / f"{item['slug']}.md").write_text(
            render_card(item, membership, duplicates), encoding="utf-8"
        )

    index_path = ROOT / "index.csv"
    fields = ["slug", "name", "membership", "site_category", "research_family", "research_role", "status", "timeframes", "source_url", "card_path", "duplicate_name"]
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item, membership in selected:
            family, role = research_classification(item)
            writer.writerow({
                "slug": item["slug"], "name": item["name"], "membership": membership,
                "site_category": item["category"], "research_family": family,
                "research_role": role, "status": "Idea",
                "timeframes": ",".join(item["timeframes"]),
                "source_url": f"{SITE}/mnq/{item['slug']}/backtest",
                "card_path": f"cards/{item['slug']}.md",
                "duplicate_name": "yes" if item["name"] in duplicates else "no",
            })

    manifest = {
        "snapshot_date": SNAPSHOT_DATE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_asset_url": ASSET_URL,
        "source_asset_sha256": hashlib.sha256(raw).hexdigest(),
        "source_strategy_count": len(all_items),
        "ranked_card_count": len(ranked),
        "supplemental_card_count": 1,
        "total_card_count": len(selected),
        "supplemental_slug": SUPPLEMENTAL_SLUG,
        "duplicate_name_groups": duplicates,
        "selection_note": "184 entries reproduce the site's >=300 synthetic-trades display filter; RSI Mean Reversion is a published MNQ page outside that filter. Site performance metrics are excluded.",
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Downloaded PineScriptForge index JavaScript asset")
    build(parser.parse_args().bundle)
