"""Render 00_reproduction.md from outputs/00_reproduction.{csv,json}."""

from __future__ import annotations

import json

import pandas as pd

from .common import ACTIVE8, AUDIT, CODES, OUT, fr, load_daily, lock_daily_nine, max_dd, sharpe


def _fmt(v, metric: str) -> str:
    if isinstance(v, str):
        return v
    if metric in ("trades", "sessions") or "trades identiques" in metric:
        return f"{int(round(v))}"
    if metric in ("sharpe", "pf"):
        return fr(v, 4)
    return fr(v, 2)


def main() -> None:
    t = pd.read_csv(OUT / "00_reproduction.csv")
    r = json.loads((OUT / "00_reproduction.json").read_text(encoding="utf-8"))
    rows = []
    for x in t.itertuples(index=False):
        rows.append(f"| {x[0]} | {x[1]} | {_fmt(x[2], x[1])} | {_fmt(x[3], x[1])} | {_fmt(x[4], x[1])} | "
                    f"{'oui' if x[5] else '**NON**'} |")
    fails = t[~t["ok"]]

    lock = lock_daily_nine()
    alt = lock.copy()
    alt["S01"] = load_daily("S01").reindex(lock.index).fillna(0.0)
    s01_rows = []
    for label, df in (("S01 = `baseline_fixed` (lock, book, netting)", lock), ("S01 = ensemble ATR 18 (fiche 01)", alt)):
        for n, cols in (("9", CODES), ("8", ACTIVE8)):
            p = df[cols].sum(axis=1)
            s01_rows.append(f"| {label} | {n} | {fr(p.sum())} | {fr(sharpe(p), 3)} | {fr(max_dd(p))} |")

    a = r["level_a"]
    s02 = {k: a[f"S02_{k}"] for k in ("full", "is", "val")}
    eng = r.get("engine", {})
    lines = [
        "# Phase 0 — Reproduction",
        "",
        "*Audit du 2026-11-15. Référence auditée : `nine_sleeve_equal_contract_reference_20261101`. "
        "Script : [phase0_reproduction.py](phase0_reproduction.py) et [phase0_engine.py](phase0_engine.py). "
        "Table brute : [outputs/00_reproduction.csv](outputs/00_reproduction.csv).*",
        "",
        "## Verdict",
        "",
        f"**{len(t) - len(fails)} contrôles sur {len(t)} passent. {len(fails)} échouent.** "
        "L'audit s'est arrêté ici le 2026-11-15, conformément à la règle 5. Il a repris après les décisions de "
        "l'utilisateur, en fin de document.",
        "",
        "Les PnL, les nombres de trades, le book, le leave-one-out et le netting se reproduisent au centime. "
        "Les échecs ne viennent pas des données ni du moteur. Ils viennent de deux incohérences de définition "
        "dans la documentation : décision 0.1 (la version de S01) et décision 0.2 (la convention de Sharpe de S02).",
        "",
        "## Conventions",
        "",
        "- Sharpe : √252 × moyenne / écart-type (ddof = 1) du PnL quotidien net en USD, jours sans trade inclus, "
        "sur le calendrier du fichier source de chaque sleeve. Les Sharpe publiés étant arrondis à 3 décimales, "
        "c'est la valeur arrondie qui est comparée, avec la tolérance de 0,005.",
        "- Coûts nominaux du lock : 1,25 USD de commission par côté (MNQ, MGC), 1,41 USD (MBT), 1 tick de slippage par côté.",
        "- Fenêtres : MNQ/MGC 2019-05-06 → 2026-03-19 (IS ≤ 2023-12-31), MBT 2021-05-03 → 2026-10-02 "
        "(IS ≤ 2025-05-28), S01 2019-05-05 → 2026-03-20 (IS ≤ 2024-02-27), book 2024-01-02 → 2026-03-19.",
        "- Niveau A : métriques recalculées depuis les fichiers gelés. Niveau B : ré-exécution des sleeves depuis "
        "les barres 1 minute avec les moteurs du repo, comparée trade par trade. Niveau C : moteur de netting "
        "`src/portfolio/aggregator.py`, barres coupées au 2026-03-19.",
        "",
        "## Table attendu / obtenu / écart",
        "",
        "| Élément | Métrique | Attendu | Obtenu | Écart | OK |",
        "|---|---|---:|---:|---:|---|",
        *rows,
        "",
        "## Écart 1 — le S01 du book n'est pas le S01 de la fiche 01 (bloquant)",
        "",
        "Les chiffres de la fiche 01 (701 trades, 19 416,50 USD, Sharpe 1,156) se reproduisent. Mais le book "
        "n'utilise pas cette sleeve. Trois ledgers S01 coexistent dans le repo :",
        "",
        "| Source | Définition | Trades | PnL total | Utilisée par |",
        "|---|---|---:|---:|---|",
        "| `research/signal_sheet_01_atr18/trades.csv` | vote de 18 modèles : ATR {14, 16, 18} × q_low {20, 25, 30} × q_high {80, 85} | 701 | 19 416,50 | fiche 01 |",
        "| `export/mnq_orb_vwap_atr_retained/trades.csv` | `baseline_fixed` : ATR 14, q_low {20, 25, 30} × q_high {90, 95} | 823 | 18 537,50 | étude de netting, règle de netting du 15/11 |",
        "| `research/portfolio_integration_phase1/sleeve_trades/mnq_orb_vwap_atr.csv` | extrait 2024+ de `baseline_fixed` | 261 | 6 329,00 | locks du 01/11 et du 03/11, BOOK.md sections 1–5 |",
        "",
        f"Sur 2024-01-02 → 2026-03-19, le S01 du lock fait 6 329 USD sur 261 trades. Celui de la fiche fait "
        f"6 035 USD sur 208 trades. Les 208 trades de la fiche sont tous inclus dans les 261 du lock, qui en compte 53 de plus. "
        f"L'écart quotidien maximal atteint {fr(r['book']['s01_sources_max_abs_diff'])} USD. "
        "Le forward replay (`frozen_vote_forward_replay_20261101`) et la règle de netting reprennent eux aussi `baseline_fixed`.",
        "",
        "Effet sur le book, fenêtre commune 2024-01-02 → 2026-03-19, 564 séances :",
        "",
        "| Version de S01 | Sleeves | PnL | Sharpe | maxDD |",
        "|---|---:|---:|---:|---:|",
        *s01_rows,
        "",
        "L'effet sur le book est faible : −294 USD et Sharpe quasi inchangé. Le problème est ailleurs : la "
        "définition exécutable de S01 est ambiguë. La fiche, qui sert de spécification de réimplémentation, "
        "décrit un signal que ni le lock ni la production ne tradent.",
        "",
        "## Écart 2 — le Sharpe publié de S02 n'est pas un Sharpe en USD",
        "",
        "Le `summary.json` de S02 a été produit par `compute_metrics` du moteur de base. Ce moteur calcule le Sharpe "
        "sur le rendement quotidien d'un compte de 50 000 USD qui capitalise (`daily_pnl / equity_{t−1}`). "
        "S03 à S09 utilisent `metric_block`, qui calcule le Sharpe sur le PnL en USD. Les deux conventions "
        "divergent dès que l'equity s'éloigne de 50 000 USD.",
        "",
        "| S02 | PnL | Sharpe publié (rendements) | Sharpe en USD (convention de l'audit) |",
        "|---|---:|---:|---:|",
        f"| IS 2019-05-06 → 2023-12-31 | {fr(s02['is']['pnl'])} | 0,831 | {fr(s02['is']['sharpe'], 3)} |",
        f"| Validation 2024-01-01 → 2026-03-19 | {fr(s02['val']['pnl'])} | 0,971 | {fr(s02['val']['sharpe'], 3)} |",
        f"| Total | {fr(s02['full']['pnl'])} | 0,873 | {fr(s02['full']['sharpe'], 3)} |",
        "",
        "Les Sharpe en rendements de S03 à S09 seraient aussi plus élevés (de +0,006 à +0,037). Seul S02 publie "
        "cette convention-là. Le PnL et les trades de S02 sont exacts.",
        "",
        "## Points secondaires, non bloquants",
        "",
        f"- **Calendrier de S01.** `signal_sheet_01_atr18/daily_results.csv` contient 2 143 lignes, dont 355 dimanches "
        f"sans séance RTH. Le Sharpe publié de 1,156 est calculé avec ces zéros. Sur le calendrier MNQ de S02 "
        f"(1 747 séances), le même PnL donne un Sharpe de {fr(a['S01_on_S02_calendar']['sharpe'], 3)}. "
        "La sleeve n'a aucun trade un dimanche : seuls le dénominateur et la convention changent.",
        "- **Hash périmé.** `signal_sheet_02_cluster/summary.json` référence un hash du manifeste `volatility_breakout_42.json` "
        "(`a1cfe825…`) différent du hash actuel (`e3e912ad…`), qui est celui du lock. Le manifeste a été réécrit après "
        "l'exécution, quand le bloc `execution_results` y a été ajouté. Le lock pointe vers la bonne version.",
        f"- **Netting.** 31 151,96 USD en séparé et 31 179,96 USD en agrégé (cap de 5 lots, jamais atteint : la position "
        "maximale théorique du book est de 3,8 lots : 0,6 MNQ + 0,2 MGC + 3 MBT). Les deux chiffres sont reproduits à 0,04 USD près, ce qui n'est qu'un arrondi "
        "de la publication.",
        "",
        "## Ré-exécution depuis les barres brutes (niveau B)",
        "",
    ]
    if eng:
        lines += ["| Ledger | Trades gelés | Trades ré-exécutés | Seulement gelés | Seulement ré-exécutés | ΔPnL absolu max (trades appariés) |",
                  "|---|---:|---:|---:|---:|---:|"]
        for k, v in eng.items():
            lines.append(f"| {k} | {v['frozen_trades']} | {v['engine_trades']} | {v['only_frozen']} | {v['only_engine']} | "
                         f"{fr(v['max_abs_pnl_diff_matched'])} |")
    else:
        lines.append("Non exécuté dans ce run (`--no-engine`).")
    lines += [
        "",
        "## Décisions prises le 2026-11-15",
        "",
        "**0.1 — S01 de référence : l'ensemble ATR 18 de la fiche 01** (ATR {14, 16, 18} × q_low {20, 25, 30} × "
        "q_high {80, 85}, 701 trades), décision de l'utilisateur. Conséquences :",
        "",
        "- toutes les phases suivantes utilisent `research/signal_sheet_01_atr18` pour S01 ;",
        "- **écart à corriger hors audit** : les locks `nine_sleeve_…_20261101` et `eight_sleeve_…_20261103`, la règle "
        "de netting `netting_rule_20261115.json`, l'étude de netting et le forward replay contiennent `baseline_fixed`. "
        "Leurs chiffres de book ne décrivent donc pas le book de référence. La correction passe par un nouveau lock "
        "versionné ; l'audit n'y touche pas (règle 2) ;",
        "- les chiffres « attendus » du book et du leave-one-out de la table ci-dessus restent ceux du lock publié. "
        "La phase 1 donne les chiffres du book de référence.",
        "",
        "**0.2 — Convention de Sharpe : PnL quotidien en USD pour toutes les sleeves** (règle 4 de la mission). "
        "Le 0,873 de la fiche 02 est un Sharpe en rendements ; en USD, il vaut 0,848.",
        "",
        "**0.3 — Nouveau lock versionné, sur instruction explicite de l'utilisateur** (« fais le nouveau lock versionné », "
        "2026-11-15). C'est une dérogation à la règle 2 de la mission, demandée par l'utilisateur. Contenu :",
        "",
        "- `research/locked_eight_sleeve_portfolio_20261115/` (`eight_sleeve_equal_contract_reference_20261115`) : "
        "S01 = ensemble ATR 18 et calendrier union. Rien d'autre ne change ;",
        "- `strategy_registry/execution/netting_rule_20261115_r2.json` : même logique de netting, appliquée au nouveau lock ;",
        "- le registre `forward_monitoring`, `ACTIVE_EIGHT_SLEEVES.md`, des notes datées dans BOOK.md et la fiche 01, "
        "et `SUPERSEDED.md` dans le lock du 03/11 ;",
        "- les tests `tests/test_locked_eight_sleeve_portfolio_20261115.py` ; le test de registre de l'ancien lock "
        "est remplacé par un test « référence remplacée » ;",
        "- le lock ne calcule pas S01 ATR 18 sur 2026-03-20 → 2026-10-02, fenêtre réservée par l'option (O) du "
        "pré-enregistrement.",
    ]
    (AUDIT / "00_reproduction.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
