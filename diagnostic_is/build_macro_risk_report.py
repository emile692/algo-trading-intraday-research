"""Build the French IS-only macro/risk diagnostic report from frozen CSV outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / "diagnostic_is" / "risk_tables"
REPORT = ROOT / "diagnostic_is" / "macro_risk_report.md"


def md(frame: pd.DataFrame) -> str:
    x = frame.copy().fillna("—")
    columns = [str(c) for c in x.columns]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in x.astype(str).itertuples(index=False, name=None):
        rows.append("| " + " | ".join(v.replace("|", "\\|") for v in values) + " |")
    return "\n".join(rows)


def f2(value: float) -> str:
    return f"{value:,.2f}"


def coverage_table(cov: pd.DataFrame, col: str) -> pd.DataFrame:
    return cov[cov[col].gt(0)].pivot(index="year", columns="event", values=col).fillna(0).astype(int).reset_index()


def main() -> None:
    a = pd.read_csv(RISK / "signal_a_macro_risk_trades.csv")
    b = pd.read_csv(RISK / "signal_b_macro_risk_trades.csv")
    tests = pd.read_csv(RISK / "all_36_tests_holm_R.csv")
    metrics = pd.read_csv(RISK / "bucket_metrics_R_primary_usd_secondary.csv")
    cov = pd.read_csv(RISK / "official_macro_coverage_by_year.csv")
    tails = pd.read_csv(RISK / "A_NQ_worst10_each_bucket_R_details.csv")
    concentration = pd.read_csv(RISK / "A_NQ_worst10_each_bucket_R_concentration.csv")
    b0a = pd.read_csv(RISK / "B0a_liquidity_and_pnl_yearly_R.csv")
    b0d = pd.read_csv(RISK / "B0d_roll_vs_outside_R.csv")
    b0b_dir = ROOT / "diagnostic_is" / "gc_b0b"
    b0b = pd.read_csv(b0b_dir / "B0b_version_comparison.csv")
    b0b_roll = pd.read_csv(b0b_dir / "B0b_gc_volume_roll_breakdown.csv")
    b0b_quality = pd.read_csv(b0b_dir / "B0b_dataset_quality_sensitivity.csv")
    b0b_thinning = pd.read_csv(b0b_dir / "B6_B0b_random_thinning_summary.csv")
    nq = a[a.asset.eq("NQ") & a.net_R.notna()]

    top = tests[tests.estimable].nsmallest(8, "p_value_R")[
        ["signal", "variable", "bucket_a", "bucket_b", "mean_net_R_difference_a_minus_b", "p_value_R", "holm_p_R"]
    ].copy()
    for col in ("mean_net_R_difference_a_minus_b", "p_value_R", "holm_p_R"):
        top[col] = top[col].map(lambda z: f"{z:.4f}")

    a1 = metrics[metrics.variable.eq("A1_macro_day")][
        ["bucket", "trades", "gross_R", "cost_R", "net_R", "mean_net_R", "gross_usd", "cost_usd", "net_usd"]
    ].copy()
    b1 = metrics[metrics.variable.eq("B1_announcement_bar")][
        ["bucket", "trades", "gross_R", "cost_R", "net_R", "mean_net_R", "gross_usd", "cost_usd", "net_usd"]
    ].copy()
    for frame in (a1, b1):
        for col in frame.columns[2:]:
            frame[col] = frame[col].map(lambda z: f"{z:.3f}" if "usd" not in col else f"{z:,.2f}")

    conc = concentration.copy()
    for col in conc.columns[2:]:
        conc[col] = conc[col].map(lambda z: "—" if pd.isna(z) else f"{z:.3f}")

    tail_show = tails[[
        "bucket", "session_date", "exit_reason", "loss_net_R", "net_usd", "mae_by_5m_R", "mae_by_10m_R",
        "mae_by_20m_R", "mae_peak_R", "mae_peak_time", "vix_t_minus_1", "vix_over_vix3m_t_minus_1",
        "first20m_gap_path", "first20m_gap_recovery_fraction", "macro_events",
    ]].copy()
    tail_show = tail_show.rename(columns={
        "loss_net_R": "net_R", "net_usd": "net_$", "mae_by_5m_R": "MAE_5m_R", "mae_by_10m_R": "MAE_10m_R",
        "mae_by_20m_R": "MAE_20m_R", "mae_peak_R": "MAE_pic_R", "mae_peak_time": "heure_pic_MAE",
        "vix_t_minus_1": "VIX_t-1", "vix_over_vix3m_t_minus_1": "VIX/VIX3M_t-1",
        "first20m_gap_path": "trajet_gap_20m", "first20m_gap_recovery_fraction": "fraction_reprise_20m",
        "macro_events": "macro",
    })
    for col in ["net_R", "MAE_5m_R", "MAE_10m_R", "MAE_20m_R", "MAE_pic_R", "VIX_t-1", "VIX/VIX3M_t-1", "fraction_reprise_20m"]:
        tail_show[col] = tail_show[col].map(lambda z: f"{z:.3f}")
    tail_show["net_$"] = tail_show["net_$"].map(lambda z: f"{z:,.2f}")
    tail_show["heure_pic_MAE"] = pd.to_datetime(tail_show.heure_pic_MAE, utc=True).dt.tz_convert("America/New_York").dt.strftime("%H:%M")

    liq = b0a.copy()
    liq["pct_volume_le_1"] = (100 * liq.pct_volume_le_1).map(lambda z: f"{z:.2f}%")
    liq["median_bar_volume"] = liq.median_bar_volume.map(lambda z: f"{z:,.1f}")
    liq["median_daily_volume"] = liq.median_daily_volume.map(lambda z: f"{z:,.1f}")
    for col in ["gross_usd", "cost_usd", "net_usd"]:
        liq[col] = liq[col].map(lambda z: f"{z:,.2f}")
    for col in ["gross_R", "cost_R", "net_R", "mean_net_R"]:
        liq[col] = liq[col].map(lambda z: f"{z:.3f}")
    liq = liq[["year", "median_bar_volume", "pct_volume_le_1", "median_daily_volume", "trades", "gross_usd", "cost_usd", "net_usd", "gross_R", "cost_R", "net_R"]]

    roll = b0d.copy()
    for col in ["gross_usd", "cost_usd", "net_usd"]:
        roll[col] = roll[col].map(lambda z: f"{z:,.2f}")
    for col in ["gross_R", "cost_R", "net_R", "mean_net_R"]:
        roll[col] = roll[col].map(lambda z: f"{z:.3f}")

    b0b_show = b0b.copy()
    for col in ["gross_usd", "cost_usd", "net_usd"]:
        b0b_show[col] = b0b_show[col].map(lambda z: f"{z:,.2f}")
    for col in ["gross_R", "cost_R", "net_R", "mean_net_R", "event_overlap_vs_version", "executable_overlap_vs_version"]:
        b0b_show[col] = b0b_show[col].map(lambda z: f"{z:.4f}")
    b0b_roll_show = b0b_roll.copy()
    for col in ["gross_usd", "cost_usd", "net_usd"]:
        b0b_roll_show[col] = b0b_roll_show[col].map(lambda z: f"{z:,.2f}")
    for col in ["gross_R", "cost_R", "net_R"]:
        b0b_roll_show[col] = b0b_roll_show[col].map(lambda z: f"{z:.3f}")
    quality_show = b0b_quality.copy()
    for col in quality_show.columns[2:]:
        quality_show[col] = quality_show[col].map(lambda z: f"{z:,.3f}")
    thinning_show = b0b_thinning[["candidate_trades", "price_only_trades", "candidate_sharpe", "candidate_expectancy", "sharpe_percentile", "expectancy_percentile"]].copy()
    for col in thinning_show.columns[2:]:
        thinning_show[col] = thinning_show[col].map(lambda z: f"{z:.4f}")

    a_sum = nq[["gross_R", "cost_R", "net_R", "gross_pnl", "cost", "net_pnl"]].sum()
    b_sum = b[["gross_R", "cost_R", "net_R", "gross_pnl", "cost", "net_pnl"]].sum()
    report = f"""# Diagnostic macro et unités de risque — IS uniquement

## Résumé

- Bornes verrouillées : A/NQ du 2010-06-07 au 2024-02-22 ; B/MGC du 2015-01-02 au 2023-03-16. Aucune statistique post-IS.
- `R = PnL $ / (ATR14 Wilder de la séance précédente × valeur du point)` ; 1 946/1 954 trades NQ ont un ATR disponible après warm-up, contre 1 333/1 333 pour MGC.
- A/NQ : brut {a_sum.gross_R:.2f} R, coûts {a_sum.cost_R:.2f} R, net {a_sum.net_R:.2f} R ; secondaire : brut ${a_sum.gross_pnl:,.2f}, coûts ${a_sum.cost:,.2f}, net ${a_sum.net_pnl:,.2f}.
- B/MGC : brut {b_sum.gross_R:.2f} R, coûts {b_sum.cost_R:.2f} R, net {b_sum.net_R:.2f} R ; secondaire : brut ${b_sum.gross_pnl:,.2f}, coûts ${b_sum.cost:,.2f}, net ${b_sum.net_pnl:,.2f}.
- 36 hypothèses planifiées (34 anciennes + A1 + B1), 35 estimables ; Holm utilise la multiplicité conservatrice 36. Aucun rejet à 5 %.
- A1 ne sépare pas les pertes : différence macro moins hors macro = +0,0028 R/trade, p=0,8926, Holm=1.
- B1 est non estimable : zéro climax dans la fenêtre d’annonce parmi 1 333 trades ; la séance MGC exclut par définition la fenêtre FOMC de 14:00.
- La queue A reste concentrée, mais moins qu’en dollars : les dix pires R expliquent 39,7 % de la perte nette R du bucket NQ [0,50–0,75).
- B0d exclut un artefact de roll dominant : 76 trades roll ±2 séances, net -1,82 R/-$266, contre -45,70 R/-$6 496,50 hors roll.
- B0b utilise les dix contrats GC les plus liquides pour 18,7575 $ : 5 137 924 lignes, un seul appel payant, sans clé persistée.

## Calendriers officiels parsés

Les lignes CPI et Employment Situation sont extraites des pages annuelles BLS. Les dates FOMC sont extraites des liens HTML officiels `monetaryYYYYMMDDa.htm` des policy statements de la Fed ; aucune date n’est codée en dur. Les 9 dates de 2010, 9 de 2019 et 12 de 2020 incluent donc les statements non programmés effectivement publiés. A1 est strictement le jour de publication.

### Couverture dans A

{md(coverage_table(cov, 'signal_A_IS_dates'))}

### Couverture dans B

{md(coverage_table(cov, 'signal_B_IS_dates'))}

Le fichier de couverture complet conserve aussi la première et la dernière date officielle par année et les comptes des pages sources.

## Contrastes en R et A1/B1

Les dollars sont secondaires et descriptifs : ils ne créent pas une seconde famille de tests.

### Huit plus petites p-values brutes

{md(top)}

La plus petite p-value brute est A4, reprise haute contre basse, p=0,0184 ; elle devient 0,6611 après Holm. Aucun résultat n’est compatible avec un état stable identifié au seuil familial de 5 %.

### A1 — jour CPI, NFP ou FOMC

{md(a1)}

### B1 — barre d’annonce

{md(b1)}

B1 ne possède que le bucket `outside`. Ce n’est pas un zéro de performance filtrée mais une absence d’observation dans la cellule d’annonce ; aucun contraste ni performance contrefactuelle n’est calculé.

## Queue de A/NQ

Les dix pires trades sont choisis séparément dans chaque bucket sur `net_R`. La MAE est reconstruite barre par barre ; son maximum concorde avec le ledger à moins de 1e-12 dollar. Une fraction de reprise positive signifie reprise du gap, négative extension.

### Concentration

{md(conc)}

Pour [0,75–1,00), la part de 2,087 signifie que les dix pires cumulent plus de deux fois la perte nette R du bucket, les autres trades compensant une partie de cette queue. Pour ≥1,00, le bucket est globalement positif en R, donc aucune « part de perte nette » n’est définie.

### Description des 40 trades

{md(tail_show)}

Lecture synthétique : 27 des 40 queues prolongent encore le gap durant les 20 premières minutes ; 27 sortent au stop et 13 au temps. Deux seulement sont des jours macro, tous deux dans [0,25–0,50). La queue [0,50–0,75) n’est donc pas une queue d’annonces macro.

## B0a — liquidité MGC et PnL annuel

{md(liq)}

La médiane par barre passe de 7 contrats en 2015 à 387 en 2020, et le volume journalier médian de 652,5 à 29 225. Pourtant les pires années nettes en R incluent 2019 (-12,04 R) et 2020 (-6,43 R) : la seule illiquidité des premières années n’explique pas monotoniquement la perte. Le constat robuste est plutôt un brut quasi nul (-3,39 R) absorbé par 44,13 R de coûts.

## B0d — roll ±2 séances contre hors roll

{md(roll)}

Les fenêtres de roll ne représentent que 5,7 % des trades et 3,8 % de la perte nette R. Elles ont même un brut positif (+0,78 R). Le roll n’explique donc pas la perte canonique.

## B0b — volume GC agrégé, exécution MGC inchangée

Le volume est la somme, minute par minute, des rangs continus `GC.v.0` à `GC.v.9`, puis agrégé en barres de cinq minutes. Le RVOL GC est causal : volume courant divisé par la médiane de la même tranche sur les vingt séances antérieures. Les prix, confirmations, fills, coûts et PnL restent ceux de MGC.

{md(b0b_show)}

Le volume GC conserve 1 234 des 1 534 événements bruts MGC (80,44 %) et 1 063 des 1 333 trades exécutables (79,74 %). Inversement, 92,99 % des événements GC et 92,52 % de ses trades sont communs avec MGC. Le point décisif est l’espérance : −0,03630 R/trade avec GC contre −0,03565 avec MGC. Le remplacement de la mesure de volume ne restaure donc aucun edge.

### B0b et roll GC

{md(b0b_roll_show)}

### Qualité Databento

{md(quality_show)}

Treize dates GLBX.MDP3 sont marquées `degraded`, mais six trades seulement sont touchés. Leur retrait laisse B0b à −41,46 R et −5 859,50 $, sans changement de conclusion.

### B6 — random thinning de B0b

{md(thinning_show)}

Le Sharpe de B0b est au percentile 38,6 % des échantillons de même taille et son expectancy au percentile 20,0 %. La sélection GC n’est pas supérieure à un amincissement aléatoire du même univers de prix.

## Acquisition Databento GC exécutée

- Dataset : `GLBX.MDP3`.
- Symboles : `GC.v.0` à `GC.v.9`, `stype_in=continuous`, afin de rester sous le crédit disponible tout en agrégeant les dix contrats les plus liquides.
- Schéma : `ohlcv-1m`, puis agrégation 5 minutes identique à MGC.
- Fenêtre API : début inclusif `2015-01-02T00:00:00Z`, fin exclusive `2023-03-17T00:00:00Z`.
- Coût estimé juste avant acquisition : **18,757453262806 $**, sous le garde dur de 20 $.
- Résultat : 5 137 924 lignes, 110 identifiants d’instrument, aucun doublon contrat-minute ; un seul appel `get_range`.
- SHA-256 brut : `262ebb63cd9b9c519d9e9c1112ab304d085c5c1370d969f7be176dce12a3bc3b`.
- SHA-256 agrégat 5 minutes : `bc94abc8950d926a0099d79eb7ee80e0c3936e96b1832e0a97d9de777f042a6e`.

## Conclusions reformulées

- **Signal A — (c2) edge nul avec queue gauche concentrée.** Le brut normalisé est négatif (-21,49 R), aucun des 35 contrastes estimables ne survit à Holm, A1 est nul, mais les dix pires R absorbent environ 40 % de la perte nette dans les deux premiers buckets et plus de 200 % dans [0,75–1,00). Il existe une queue, pas un état exploitable démontré.
- **Signal B — (c1) edge nul et pertes diffuses.** B0b invalide l’explication (b) : malgré environ 20 % d’événements différents, GC donne pratiquement la même espérance par trade que MGC. Le brut reste proche de zéro/négatif, les coûts dominent, les pertes subsistent hors roll et après exclusion des dates dégradées, et le random thinning n’indique aucune sélection particulière.

Aucune règle de trading ni performance filtrée n’est produite.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
