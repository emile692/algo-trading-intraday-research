# Phase 4 — Holdout

*Exécuté une seule fois par [phase4b_s01_oneshot.py](phase4b_s01_oneshot.py), selon le [pré-enregistrement validé](preregistration.md). Sorties : [04b_preread.json](outputs/04b_preread.json), écrit avant tout chargement de prix, et [04b_result.json](outputs/04b_result.json).*

## (O) — S01 ATR 18, 2026-03-20 → 2026-10-02 (partiellement contaminé)

- **H** = 137 séances MNQ (2026-03-23 → 2026-10-02), comptées sur les timestamps seuls.
- **Seuil pré-enregistré** : p5 du PnL à H = **−617,50 USD** (médiane 1 455,00, p95 3 817,00).
- **Gate technique** : 701 trades rejoués jusqu'au 2026-03-19 pour 701 gelés ; 0 seulement gelés, 0 seulement rejoués, écart PnL max 0,00 USD. **Réussi.**

| Critère (tel qu'écrit) | Valeur | Verdict |
|---|---:|---|
| PnL cumulé à H < p5 ⇒ ÉCHEC | −390,00 USD vs −617,50 | **COHÉRENT** |

Statistiques reportées, sans critère :

- trades : 11 ; Sharpe (√252, 137 séances) : −0,756 ; maxDD : −840,50 USD ; pire jour : −245,00 USD ;
- pour information, `baseline_fixed` sur la même fenêtre (déjà lu le 2026-11-01) : 503,00 USD.

Ce résultat ne sert à aucun retuning. Il alimente le statut de S01 en phase 6.

## (P) — Suivi prospectif après le 2026-11-03

Non lu : les données ne sont pas sur disque. La lecture se fait avec `forward_check.py`, une fois par horizon (20, 60, 120, 250 séances). Première lecture possible vers le 2026-12-03, après téléchargement des données.
