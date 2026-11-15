"""Phase 4a — criteria tables and preregistration, written BEFORE any holdout / forward data is loaded.

Bootstrap: Politis-Romano stationary bootstrap of each series' daily net PnL on its history up to 2026-03-19
(MNQ/MGC sleeves from 2019-05-06, MBT sleeves from 2021-05-03, books on the union calendar from 2021-05-03),
10 000 paths of 260 sessions, mean block 20 (criterion) and 5 (reported only), seed 20261115 + 4.
Output: outputs/04_criteria.csv (every horizon 1..260), preregistration.md, preregistration.sha256.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from .common import ACTIVE8, AUDIT, CODES, INSTRUMENT, NAMES, OUT, ROOT, SEED, fr, sha256
from .phase1_calendar import instrument_calendars
from .stats import stationary_indices

N_PATHS, MAX_H, BLOCK_CRIT, BLOCK_ALT = 10_000, 260, 20, 5
LOOKS = (20, 60, 120, 250)
HIST_END = "2026-03-19"
START = {"MNQ": "2019-05-06", "MGC": "2019-05-06", "MBT": "2021-05-03", "BOOK": "2021-05-03"}


def histories() -> dict[str, pd.Series]:
    p = pd.read_csv(OUT / "daily_pnl_union.csv", parse_dates=["session_date"]).set_index("session_date")
    cals = instrument_calendars()
    out = {}
    for code in CODES:
        inst = INSTRUMENT[code]
        s = p[code].reindex(cals[inst])
        out[code] = s[(s.index >= START[inst]) & (s.index <= HIST_END)]
    b = p[(p.index >= START["BOOK"]) & (p.index <= HIST_END)]
    out["Book 8"] = b[ACTIVE8].sum(axis=1)
    out["Book 9"] = b[CODES].sum(axis=1)
    for k, v in out.items():
        if v.isna().any():
            raise RuntimeError(k)
    return out


def path_stats(x: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = x[idx]
    eq = np.cumsum(m, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.zeros((len(m), 1)), eq], axis=1), axis=1)[:, 1:]
    dd = np.minimum.accumulate(eq - peak, axis=1)
    return eq, dd


def main() -> dict:
    rng = np.random.default_rng(SEED + 4)
    hist = histories()
    rows, alarm = [], []
    for name, s in hist.items():
        x = s.to_numpy(float)
        for block in (BLOCK_CRIT, BLOCK_ALT):
            idx = stationary_indices(len(x), N_PATHS, block, rng, length=MAX_H)
            eq, dd = path_stats(x, idx)
            q = {k: np.percentile(v, [5, 50, 95], axis=0) for k, v in (("pnl", eq), ("mdd", dd))}
            for h in range(1, MAX_H + 1):
                rows.append({"series": name, "block": block, "H": h,
                             "pnl_p05": q["pnl"][0, h - 1], "pnl_p50": q["pnl"][1, h - 1], "pnl_p95": q["pnl"][2, h - 1],
                             "mdd_p05": q["mdd"][0, h - 1], "mdd_p50": q["mdd"][1, h - 1], "mdd_p95": q["mdd"][2, h - 1]})
            if block == BLOCK_CRIT:
                hs = np.array(LOOKS) - 1
                if name.startswith("Book"):
                    fail = ((eq[:, hs] < q["pnl"][0, hs]) | (eq[:, hs] > q["pnl"][2, hs])
                            | (dd[:, hs] < q["mdd"][0, hs]) | (dd[:, hs] > q["mdd"][2, hs]))
                    fail_low = (eq[:, hs] < q["pnl"][0, hs]) | (dd[:, hs] < q["mdd"][0, hs])
                else:
                    fail = fail_low = eq[:, hs] < q["pnl"][0, hs]
                alarm.append({"series": name, "history_sessions": len(x),
                              **{f"p_fail_H{h}": float(fail[:, i].mean()) for i, h in enumerate(LOOKS)},
                              "p_any_look": float(fail.any(axis=1).mean()),
                              "p_any_look_low_side_only": float(fail_low.any(axis=1).mean())})
    crit = pd.DataFrame(rows)
    crit.to_csv(OUT / "04_criteria.csv", index=False, float_format="%.4f")
    al = pd.DataFrame(alarm)
    al.to_csv(OUT / "04_false_failure_rates.csv", index=False)
    write_prereg(crit, al, hist)
    return {"criteria_sha256": sha256(OUT / "04_criteria.csv")}


def write_prereg(crit: pd.DataFrame, al: pd.DataFrame, hist: dict) -> None:
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    crit_hash = sha256(OUT / "04_criteria.csv")
    c = crit[crit["block"] == BLOCK_CRIT].set_index(["series", "H"])
    tab = ["| Série | Historique (séances) | " + " | ".join(f"p5 PnL H={h}" for h in LOOKS) + " |",
           "|---|---:|" + "---:|" * len(LOOKS)]
    for name in hist:
        tab.append(f"| {name} | {len(hist[name])} | " + " | ".join(fr(c.loc[(name, h), 'pnl_p05'], 0) for h in LOOKS) + " |")
    btab = ["| Book | H | PnL p5 | PnL p95 | maxDD p5 | maxDD p95 |", "|---|---:|---:|---:|---:|---:|"]
    for name in ("Book 8", "Book 9"):
        for h in LOOKS:
            r = c.loc[(name, h)]
            btab.append(f"| {name} | {h} | {fr(r.pnl_p05, 0)} | {fr(r.pnl_p95, 0)} | {fr(r.mdd_p05, 0)} | {fr(r.mdd_p95, 0)} |")
    atab = ["| Série | P(échec) H=20 | H=60 | H=120 | H=250 | P(au moins un échec sur 4 lectures) |",
            "|---|---:|---:|---:|---:|---:|"]
    for r in al.itertuples(index=False):
        atab.append(f"| {r.series} | {fr(100 * r.p_fail_H20, 1)} % | {fr(100 * r.p_fail_H60, 1)} % | {fr(100 * r.p_fail_H120, 1)} % | "
                    f"{fr(100 * r.p_fail_H250, 1)} % | {fr(100 * r.p_any_look, 1)} % |")
    lock = ROOT / "research" / "locked_nine_sleeve_portfolio_20261101" / "locked_nine_sleeve_config.json"
    import json
    cfg = json.loads(lock.read_text(encoding="utf-8"))
    src = cfg["sleeve_sources"]
    s01_files = ["research/signal_sheet_01_atr18/thresholds_is.csv", "research/signal_sheet_01_atr18/summary.json",
                 "scripts/rebuild_signal_sheet_01_atr18.py", "src/strategy/orb.py"]
    conf = [f"| S01 | `{NAMES['S01']}` | ensemble ATR 18 de la fiche 01 | " +
            "<br>".join(f"`{f}` `{sha256(ROOT / f)[:16]}…`" for f in s01_files) + " |"]
    for code in CODES[1:]:
        m = src[NAMES[code]]
        cur = sha256(ROOT / m["ensemble_manifest"])
        if cur != m["ensemble_manifest_sha256"]:
            raise RuntimeError(f"{code}: manifest hash differs from the lock")
        conf.append(f"| {code} | `{NAMES[code]}` | vote gelé | `{m['ensemble_manifest']}` `{cur[:16]}…` |")
    lines = [
        "# Pré-enregistrement — validation hors échantillon de la référence du 01/11/2026",
        "",
        f"- **Rédigé le** : {now}",
        "- **Auteur** : audit `audit_20261115`, sur instruction de l'utilisateur. Statut : **EN ATTENTE DE VALIDATION**.",
        f"- **Table des seuils** : [`outputs/04_criteria.csv`](outputs/04_criteria.csv), sha256 `{crit_hash}`.",
        "- **Empreinte de ce fichier** : `preregistration.sha256`. Toute modification après validation invalide le "
        "pré-enregistrement.",
        "",
        "## 1. Données évaluées",
        "",
        "La phase 3 a établi qu'**aucune donnée vierge n'existe sur disque**. Ce pré-enregistrement porte donc sur deux "
        "évaluations :",
        "",
        "- **(P) Suivi prospectif, évaluation principale.** Les séances postérieures au 2026-11-03, pas encore "
        "téléchargées. Les critères sont lus aux horizons H = 20, 60, 120 et 250 séances. Chaque lecture se fait "
        "**une seule fois**, quand l'horizon est atteint, et ne se refait jamais.",
        "  - Sleeve : H compte les séances de son instrument, à partir de la première séance après le 2026-11-03.",
        "  - Book : H compte les séances du calendrier union.",
        "- **(O) Option, à valider séparément : S01 one-shot sur 2026-03-20 → 2026-10-02.** Pour S01 seul. "
        "« Partiellement contaminé » : la version `baseline_fixed`, qui partage l'entrée, la sortie et le sizing, a "
        "déjà été lue sur cette fenêtre. H est le nombre de séances MNQ de la fenêtre. Le seuil est lu dans la table "
        "à cet H, avant tout calcul de PnL.",
        "",
        "Ni (P) ni (O) n'est chargé avant votre validation écrite.",
        "",
        "## 2. Configuration testée",
        "",
        "Référence `nine_sleeve_equal_contract_reference_20261101`, avec un S01 remplacé : **S01 est l'ensemble ATR 18 "
        "de la fiche 01** (décision de l'utilisateur du 2026-11-15), et non `baseline_fixed` que contient le lock. "
        "Une copie entière par sleeve. Exécution **séparée** : chaque sleeve est rejouée seule, son PnL est reporté "
        "seul, et le PnL du book est la somme des sleeves. Le netting n'est pas testé ici. Coûts nominaux : 1,25 USD "
        "par côté (MNQ, MGC), 1,41 USD (MBT), 1 tick de slippage par côté.",
        "",
        "| Code | Sleeve | Définition | Fichiers gelés (sha256) |",
        "|---|---|---|---|",
        *conf,
        "",
        "Exigences d'exécution :",
        "",
        "- S01 utilise les seuils figés de `thresholds_is.csv`. Ils ne sont **pas** recalibrés : le découpage 70/30 "
        "du script de reconstruction se déplacerait sur un fichier plus long.",
        "- S02 à S09 utilisent les cellules et le vote de leurs manifestes, avec le moteur du repo, sans changement.",
        "- Books évalués : **Book 8** (S01, S02, S03, S05, S06, S07, S08, S09) et **Book 9** (les mêmes, plus S04).",
        "",
        "## 3. Métriques et conventions",
        "",
        "- PnL quotidien net en USD, coûts nominaux, jours sans trade inclus.",
        "- **PnL cumulé à H** : somme des H premières séances.",
        "- **maxDD à H** : drawdown maximal clôture-à-clôture de l'equity cumulée depuis 0 sur les H premières séances "
        "(valeur ≤ 0).",
        "- Sharpe (√252), nombre de trades et pire jour : reportés, sans critère.",
        "",
        "## 4. Distribution de référence",
        "",
        "- Bootstrap stationnaire de Politis-Romano sur l'historique de chaque série jusqu'au 2026-03-19 : MNQ/MGC "
        "depuis le 2019-05-06, MBT depuis le 2021-05-03, books depuis le 2021-05-03 sur le calendrier union.",
        "- Bloc moyen **20** pour les critères. Le bloc 5 est calculé et reporté, mais pas utilisé.",
        "- 10 000 chemins de 260 séances, seed 20261119.",
        "- Les seuils de tous les H de 1 à 260 sont dans la table hachée ci-dessus. Les historiques incluent la "
        "validation réutilisée : la distribution reflète donc la performance déjà vue, optimiste pour les sleeves "
        "sélectionnées sur cette fenêtre.",
        "",
        "## 5. Critères (lus tels quels, sans réinterprétation)",
        "",
        "- **Sleeve — ÉCHEC** si son PnL cumulé à H est **strictement inférieur** au p5 de sa distribution bootstrap à H. "
        "**COHÉRENT** sinon.",
        "- **Book — ÉCHEC** si son PnL cumulé à H est hors de [p5 ; p95], ou si son maxDD à H est hors de [p5 ; p95]. "
        "Bornes incluses dans l'intervalle. Le sens est reporté : « ÉCHEC-BAS » si PnL < p5 ou maxDD < p5, "
        "« ÉCHEC-HAUT » si PnL > p95 ou maxDD > p95 (drawdown plus faible que prévu). **COHÉRENT** sinon.",
        "- Aucun critère de succès plus fort que la cohérence.",
        "",
        "Seuils de PnL aux horizons de lecture (bloc 20) :",
        "",
        *tab,
        "",
        "Seuils des books :",
        "",
        *btab,
        "",
        "## 6. Taux de fausses alarmes attendu si le modèle est vrai",
        "",
        "Ce taux est calculé sur les mêmes chemins bootstrap. Il sert à lire le verdict : avec 8 sleeves et 4 lectures, "
        "plusieurs ÉCHEC isolés sont attendus même sans dégradation.",
        "",
        *atab,
        "",
        "## 7. Engagements",
        "",
        "1. Ces résultats ne serviront **à aucun retuning** : ni paramètre de signal, ni cluster, ni seuil de vote, ni "
        "taille, ni horaire, ni coût.",
        "2. Chaque lecture est faite une fois. Si un bug apparaît après la lecture, il est documenté et l'évaluation "
        "s'arrête. Aucun recalcul.",
        "3. Un ÉCHEC n'entraîne aucune action automatique. Toute décision de statut (actif, shadow) relève de "
        "l'utilisateur et de la règle d'arrêt en vigueur (phase 5).",
        "4. Ce fichier et `04_criteria.csv` ne sont plus modifiés après validation. Leur sha256 est vérifié avant "
        "chaque lecture.",
        "",
        "## 8. Points à confirmer à la validation",
        "",
        "- **Lecture bilatérale du critère du book.** La mission dit « hors de l'intervalle 5 %–95 % ». Un PnL trop "
        "*bon* ou un drawdown trop *faible* serait donc classé ÉCHEC-HAUT. Je l'ai écrit ainsi ; dites-moi si seul "
        "le côté bas doit compter.",
        "- **Option (O)** : la lancer ou non.",
        "- **S01 en production.** La règle de netting du 15/11 trade `baseline_fixed`. Pour que (P) teste le S01 de "
        "référence, il faut calculer en forward le PnL virtuel de l'ensemble ATR 18. Je reporterai les deux, mais "
        "seul l'ATR 18 sera jugé.",
    ]
    path = AUDIT / "preregistration.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (AUDIT / "preregistration.sha256").write_text(f"{sha256(path)}  preregistration.md\n{crit_hash}  outputs/04_criteria.csv\n",
                                                encoding="utf-8")


if __name__ == "__main__":
    print(main())
