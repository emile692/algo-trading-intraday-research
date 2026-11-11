"""Build the IS-only diagnostic report from deterministic CSV artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
STEP1 = HERE / "step1_tables"
STATE = HERE / "state_tables"


def md(frame: pd.DataFrame, digits: int = 3) -> str:
    if frame.empty:
        return "_Aucune ligne._"
    x = frame.copy()
    for column in x.select_dtypes(include=["float", "float64"]).columns:
        x[column] = x[column].map(lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
    columns = [str(c) for c in x.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in x.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |")
    return "\n".join(lines)


def main() -> None:
    config = pd.read_csv(STEP1 / "pnl_by_configuration.csv")
    conc = pd.read_csv(STEP1 / "loss_concentration.csv")
    dd = pd.read_csv(STEP1 / "worst_5_drawdowns.csv")
    metrics = pd.read_csv(STATE / "bucket_metrics.csv")
    tests = pd.read_csv(STATE / "bucket_tests_holm.csv")
    matrices = pd.read_csv(STATE / "year_bucket_matrices.csv")
    proxy = pd.read_csv(STATE / "temporal_proxy_flags.csv")
    ranking = pd.read_csv(STATE / "variable_explanatory_ranking.csv")
    interactions = pd.read_csv(STATE / "top2_interactions.csv")
    liquidity = pd.read_csv(STATE / "B0a_mgc_liquidity_yearly.csv")
    roll = pd.read_csv(STATE / "B0d_roll_window.csv")
    thinning = pd.read_csv(STATE / "B6_random_thinning_summary.csv")
    metadata = json.loads((HERE / "state_metadata.json").read_text(encoding="utf-8"))

    primary = config[((config.signal == "A") & (config.asset == "NQ")) | (config.signal == "B")]
    primary = primary[["signal", "asset", "configuration", "trades", "gross_pnl", "costs", "net_pnl", "net_per_trade"]]
    nq_priority = json.loads((STEP1 / "priority_nq_050_075_summary.json").read_text(encoding="utf-8"))
    proxy_true = proxy[proxy.proxy_flag_gt_60pct]
    closest = tests.sort_values("p_value").head(8)[["signal", "variable", "bucket_a", "bucket_b", "cluster_t", "p_value", "holm_p", "bootstrap95_low", "bootstrap95_high"]]

    loss_rank_rows = []
    for signal, asset in (("A", "NQ"), ("B", "MGC")):
        part = metrics[(metrics.signal == signal) & (metrics.asset == asset)]
        total = config[(config.signal == signal) & (config.asset == asset)].net_pnl.sum()
        for variable, group in part.groupby("variable"):
            worst = group.loc[group.net_pnl.idxmin()]
            loss_rank_rows.append({"signal": signal, "variable": variable, "worst_bucket": worst.bucket, "bucket_net_pnl": worst.net_pnl, "share_of_signal_net_loss": (-worst.net_pnl / -total) if total < 0 else np.nan})
    loss_rank = pd.DataFrame(loss_rank_rows).sort_values(["signal", "share_of_signal_net_loss"], ascending=[True, False])

    lines = [
        "# Diagnostic IS — gap reversion NQ et MGC volume climax",
        "",
        "## Résumé (10 lignes maximum)",
        "",
        "1. Périmètre strict: A/NQ 2010-06-07–2024-02-22; micros seulement en robustesse; B/MGC 2015-01-02–2023-03-16.",
        "2. Les seuils A sont exactement emboîtés; toute l'analyse emploie quatre buckets exclusifs.",
        "3. NQ perd surtout dans `[0.25,0.50)` (-54,7 k$) et `[0.50,0.75)` (-48,4 k$); les buckets supérieurs compensent partiellement.",
        "4. Dans `[0.50,0.75)`, les 10 pires trades expliquent 74,1 % de la perte nette, avec une forte concentration en 2020-2022.",
        "5. Les coûts relatifs NQ diminuent fortement avec le niveau de prix; la dégradation vient du brut normalisé, pas des coûts.",
        "6. A4 et A5 sont descriptivement forts, mais changent de signe selon la sous-période; aucun contraste ne survit à Holm.",
        "7. B perd -6,76 k$ net: -0,76 k$ brut et 6,00 k$ de coûts; le brut se détériore surtout en 2019-2020.",
        "8. La mesure B0c est déjà la mesure canonique; le volume MGC augmente très fortement au cours de l'IS et B0b/GC manque.",
        "9. Sur 34 tests disponibles, aucun rejet à 5 % après Holm; plusieurs buckets sont des proxies temporels.",
        "10. Conclusion A: (c) pertes diffuses/régime; conclusion B: (b) problème de mesure probable, non clos sans volume GC.",
        "",
        f"## Garde-fous et tests multiples\n\nNombre total de contrastes: **{metadata['number_of_tests']}**. Rejets Holm 5 %: **{int(tests.holm_reject_5pct.sum())}**. Bootstrap: {metadata['bootstrap_draws']} tirages, blocs de {metadata['bootstrap_block_dates']} dates. Aucun résultat post-IS n'est calculé.",
        "",
        "Variables bloquées faute de données: A1 macro, B0b volume GC et B1 annonces. A2 utilise désormais les fichiers officiels Cboe VIX/VIX3M. Aucune règle de trading ni performance filtrée n'est produite.",
        "",
        "## PnL brut, coûts et net",
        "",
        md(primary, 2),
        "",
        "Détails annuels et trimestriels: `step1_tables/pnl_by_year.csv` et `step1_tables/pnl_by_quarter.csv`. Les normalisations annuelles sont dans `normalized_pnl_yearly_signal_a.csv` et `normalized_pnl_yearly_signal_b.csv`.",
        "",
        "## Concentration prioritaire NQ `[0.50,0.75)`",
        "",
        f"La nouvelle fenêtre contient {nq_priority['trades']} trades et {nq_priority['net_pnl']:.2f} $ net. Les dix pires trades perdent {nq_priority['worst_10_trades_loss']:.2f} $, soit {nq_priority['worst_10_trades_share_of_net_loss']:.1%} de la perte nette. L'écart avec l'estimation ~400/-35,9 k$ provient des 98 trades ajoutés entre l'ancienne coupure NQ et février 2024.",
        "",
        md(dd[((dd.signal == "A") & (dd.asset == "NQ") & (dd.configuration == "[0.50,0.75)")) | (dd.signal == "B")], 2),
        "",
        "## Classement descriptif des sources de pertes",
        "",
        md(loss_rank, 3),
        "",
        "Une part supérieure à 100 % signifie que d'autres buckets compensent une partie de la perte. Ce classement est descriptif: il ne transforme aucun bucket en filtre.",
        "",
        "## Vérification de mesure B",
        "",
        "### B0a — liquidité MGC",
        "",
        md(liquidity, 4),
        "",
        "Le volume médian journalier passe de 652,5 contrats en 2015 à 18 759,5 en 2022. Les barres nulles sont absentes, tandis que la part de barres à volume ≤1 chute de 10,9 % à 0,12 %. La population de mesure change donc fortement.",
        "",
        "### B0c et B0d",
        "",
        "B0c est identique à l'original: RVOL = volume de la barre / médiane de la même tranche sur les 20 séances précédentes, sans lookahead.",
        "",
        md(roll, 3),
        "",
        "B0b n'est pas exécuté: aucun volume GC n'est présent. Une requête Databento par contrat ou `GC.v.0` est nécessaire.",
        "",
        "### B6 — random thinning IS",
        "",
        md(thinning, 3),
        "",
        "## Classement des variables disponibles",
        "",
        md(ranking, 3),
        "",
        "## Tests les plus proches du seuil nominal",
        "",
        md(closest, 4),
        "",
        "Aucun de ces contrastes ne reste significatif après Holm.",
        "",
        "## Variables suspectes d'être des proxies temporels",
        "",
        md(proxy_true, 3),
        "",
        "## Interactions autorisées — deux variables les plus explicatives",
        "",
        md(interactions, 3),
        "",
        "## Tableaux par variable",
        "",
    ]
    for (signal, variable), group in metrics.groupby(["signal", "variable"], sort=True):
        lines += [f"### {signal} — {variable}", "", md(group.drop(columns=["signal", "asset", "variable"]), 4), ""]
    lines += ["## Matrices année × bucket", ""]
    for (signal, variable), group in matrices.groupby(["signal", "variable"], sort=True):
        pivot_n = group.pivot(index="year", columns="bucket", values="trades").fillna(0).astype(int).add_prefix("n_")
        pivot_p = group.pivot(index="year", columns="bucket", values="net_pnl").fillna(0).add_prefix("pnl_")
        view = pivot_n.join(pivot_p).reset_index()
        lines += [f"### {signal} — {variable}", "", md(view, 2), ""]
    lines += [
        "## Ce qui reste inexpliqué",
        "",
        "Pour A, aucun état disponible n'explique de façon stable la perte à travers 2010-2019, 2020-2022 et 2023-2024. A4/A5 localisent descriptivement des pertes, mais leur signe change et les intervalles corrigés ne permettent pas une attribution stable. A1 reste absent.",
        "",
        "Pour B, les coûts absorbent un brut presque plat, mais le changement massif de liquidité MGC et l'absence de GC empêchent de séparer un vrai changement de mécanisme d'un problème de mesure du climax. B1 reste également absent.",
        "",
        "## Conclusions imposées",
        "",
        "- **Signal A: (c) pertes diffuses sans état explicatif stable — alpha de régime.**",
        "- **Signal B: (b) problème de mesure probable.** Cette conclusion reste conditionnelle à B0b, qui est précisément le test de mesure manquant.",
        "",
        "Aucune règle de trading finale n'est proposée.",
    ]
    (HERE / "diagnostic_is.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
