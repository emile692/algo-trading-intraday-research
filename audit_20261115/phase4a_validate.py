"""Phase 4a — turn the hashed draft (preregistration_v1_draft.md) into the validated preregistration.

Applies, as literal text substitutions on the draft, the three decisions of the user of 2026-11-15
(book criterion low side only; option (O) launched; S01 judged on the ATR 18 ensemble only) and the
pre-declared technical gate of (O).  Runs once: refuses if preregistration.md is already VALIDATED.
No market data is loaded by this script.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from .common import AUDIT, OUT, fr, sha256

DRAFT = AUDIT / "preregistration_v1_draft.md"
FINAL = AUDIT / "preregistration.md"


def sub(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one occurrence of: {old[:80]}")
    return text.replace(old, new)


def main() -> dict:
    if "Statut : **VALIDÉ**" in FINAL.read_text(encoding="utf-8"):
        raise SystemExit("preregistration.md is already validated; it is never rewritten")
    draft_hash = (AUDIT / "preregistration_v1_draft.sha256").read_text(encoding="utf-8").split()[0]
    if sha256(DRAFT) != draft_hash:
        raise SystemExit("draft does not match its recorded hash")
    crit_hash = sha256(OUT / "04_criteria.csv")
    now = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    t = DRAFT.read_text(encoding="utf-8")

    t = sub(t, "Statut : **EN ATTENTE DE VALIDATION**.",
            f"Statut : **VALIDÉ** le {now}, avec les trois décisions de l'utilisateur du 2026-11-15 "
            f"reportées en section 8. Brouillon d'origine conservé : `preregistration_v1_draft.md` "
            f"(sha256 `{draft_hash}`).")
    t = sub(t, "- **(O) Option, à valider séparément : S01 one-shot sur 2026-03-20 → 2026-10-02.** Pour S01 seul.",
            "- **(O) S01 one-shot sur 2026-03-20 → 2026-10-02 — VALIDÉE, lancée une seule fois.** Pour S01 seul.")
    t = sub(t, "H est le nombre de séances MNQ de la fenêtre. Le seuil est lu dans la table à cet H, avant tout calcul de PnL.",
            "H est le nombre de séances MNQ de la fenêtre : dates du lundi au vendredi ayant au moins une barre "
            "entre 09:30 et 16:00 ET dans `data/processed/parquet/MNQ_c_0_1m_20190505_20261003.parquet`, comptées "
            "à partir des seuls timestamps. Le seuil est lu dans la table à cet H, avant tout calcul de PnL.\n"
            "  - **Gate technique, pré-déclaré.** S01 est rejoué sur ce fichier long avec les seuils figés de "
            "`thresholds_is.csv`. Avant tout calcul sur la fenêtre, les trades jusqu'au 2026-03-19 doivent être "
            "identiques aux 701 trades gelés (date, entrée, sortie, quantité, PnL net à 0,01 USD près). Si le gate "
            "échoue, l'évaluation s'arrête sans qu'aucune statistique de la fenêtre ne soit calculée ni affichée. "
            "L'écart est documenté, et aucune correction suivie d'une relance n'est faite sur cette fenêtre.")
    t = sub(t, "Ni (P) ni (O) n'est chargé avant votre validation écrite.",
            "(O) est exécutée par `phase4b_s01_oneshot.py`, qui refuse une seconde exécution. (P) est lue par "
            "`forward_check.py`, une seule fois par horizon.")
    t = sub(t, "Référence `nine_sleeve_equal_contract_reference_20261101`, avec un S01 remplacé :",
            "Référence `nine_sleeve_equal_contract_reference_20261101`, avec un S01 remplacé (c'est exactement "
            "le contenu du lock `eight_sleeve_equal_contract_reference_20261115`, avec S04 en plus pour le Book 9) :")
    t = sub(t, "- **Book — ÉCHEC** si son PnL cumulé à H est hors de [p5 ; p95], ou si son maxDD à H est hors de [p5 ; p95]. "
               "Bornes incluses dans l'intervalle. Le sens est reporté : « ÉCHEC-BAS » si PnL < p5 ou maxDD < p5, "
               "« ÉCHEC-HAUT » si PnL > p95 ou maxDD > p95 (drawdown plus faible que prévu). **COHÉRENT** sinon.",
            "- **Book — ÉCHEC** si son PnL cumulé à H est **strictement inférieur** au p5, ou si son maxDD à H est "
            "**strictement inférieur** (plus profond) au p5 de sa distribution à H. **COHÉRENT** sinon. Un PnL "
            "au-dessus du p95 ou un maxDD au-dessus du p95 est reporté comme « au-dessus de la fourchette », mais "
            "n'est pas un ÉCHEC (décision du 2026-11-15).\n"
            "- **S01** : seul l'ensemble ATR 18 est jugé. `baseline_fixed` est reporté à titre d'information, sans critère.")

    al = pd.read_csv(OUT / "04_false_failure_rates.csv")
    start = t.index("## 6. Taux de fausses alarmes")
    end = t.index("## 7. Engagements")
    rows = ["## 6. Taux de fausses alarmes attendu si le modèle est vrai", "",
            "Ce taux est calculé sur les mêmes chemins bootstrap, avec les critères de la section 5 (côté bas seul). "
            "Avec 8 sleeves et 4 lectures, plusieurs ÉCHEC isolés sont attendus même sans dégradation.", "",
            "| Série | P(au moins un ÉCHEC sur 4 lectures) |", "|---|---:|"]
    for r in al.itertuples(index=False):
        rows.append(f"| {r.series} | {fr(100 * r.p_any_look_low_side_only, 1)} % |")
    rows.append("")
    rows.append("Pour une sleeve, le taux est de 5 % par lecture (par construction du p5). Pour un book, il est plus élevé, "
                "puisque deux conditions (PnL et maxDD) peuvent chacune déclencher l'ÉCHEC.")
    t = t[:start] + "\n".join(rows) + "\n\n" + t[end:]

    start = t.index("## 8. Points à confirmer à la validation")
    t = t[:start] + "\n".join([
        "## 8. Décisions de validation (utilisateur, 2026-11-15)", "",
        "1. **Critère du book : côté bas seul** (« Seulement pire que prévu »).",
        "2. **Option (O) : lancée**, une seule fois, avec le gate technique de la section 1.",
        "3. **S01 en forward : seul l'ensemble ATR 18 est jugé** ; `baseline_fixed` est reporté pour information.",
        "",
        f"La table des seuils `outputs/04_criteria.csv` est inchangée (sha256 `{crit_hash}`).",
    ]) + "\n"
    FINAL.write_text(t, encoding="utf-8")
    (AUDIT / "preregistration.sha256").write_text(
        f"{sha256(FINAL)}  preregistration.md\n{crit_hash}  outputs/04_criteria.csv\n", encoding="utf-8")
    return {"preregistration_sha256": sha256(FINAL), "criteria_sha256": crit_hash, "validated_at": now}


if __name__ == "__main__":
    print(main())
