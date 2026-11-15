# Audit de la référence intraday — rapport

*Audit du 2026-11-15. Référence auditée : `nine_sleeve_equal_contract_reference_20261101`, et son exécution agrégée par instrument du 13/11. Sauf mention contraire, les chiffres sont des PnL quotidiens nets en USD, avec les coûts nominaux du lock (1,25 USD de commission par côté sur MNQ et MGC, 1,41 sur MBT, 1 tick de slippage par côté). Les Sharpe sont annualisés par √252, jours sans trade inclus. Tout se relance avec `python -m audit_20261115.run_all`. Seules exceptions : les lectures one-shot `phase4b_s01_oneshot` (déjà faite, elle refuse une relance) et `forward_check` (à venir).*

---

## 1. Synthèse

### Sleeves et statuts

La référence active est désormais `eight_sleeve_equal_contract_reference_20261115`. Ce lock a été créé pendant l'audit, sur votre instruction : S01 y est l'ensemble ATR 18 de la fiche 01, et le PnL est calculé sur le calendrier union.

| Sleeve | Instrument | Statut actuel | Statut selon la règle uniforme proposée (au 2026-11-01) | Significativité (total) | Note |
|---|---|---|---|---|---|
| S01 ORB ATR 18 | MNQ | actif | actif | distinguable de zéro | one-shot (O) COHÉRENT, mais la sleeve s'assèche (2 à 9 trades par trimestre en 2026) : ses bornes d'ATR sont en points, et le prix de MNQ a doublé (décision 1) |
| S02 vol. breakout | MNQ | actif | actif | distinguable, mais pas sur l'IS seul ; DSR 0,39 | 41 % du risque du book |
| S03 compression | MNQ | actif | actif | indistinguable | |
| S04 ATR trend flip | MNQ | shadow | shadow | indistinguable | cohérent avec la règle |
| S05 vol. breakout | MGC | actif | **revue** (shadow avec la règle initiale « R0 ou R2 ») | indistinguable | R0, la règle déjà enregistrée, déclenchée du 2026-07-28 au 08-25 |
| S06 gap continuation | MGC | actif | actif | indistinguable (t = 1,97) | |
| S07 range breakout | MBT | actif | **revue** | indistinguable | R3 le 2026-08-18 |
| S08 trend pullback | MBT | actif | **revue** | indistinguable | R3 de juin à août 2026 |
| S09 vol. breakout | MBT | actif | **revue** | distinguable, grâce à l'IS seulement | validation nulle (−73 USD), R1 dès le 2025-08-13 |

### Chiffres de tête, sous contrainte Topstep

Exécution agrégée, règle actuelle, palier XFA dynamique (2 lots, puis 3, puis 5 selon le solde), 1 MBT = 1 lot, book à 8 sleeves :

| Fenêtre | PnL | Sharpe | maxDD | Pire jour | Séances réduites par la limite |
|---|---:|---:|---:|---:|---:|
| IS 2021-05-03 → 2023-12-29 | **29 142** | 2,32 | −2 567 | −967 | 21 / 689 |
| Validation réutilisée 2024-01-02 → 2026-03-19 | **29 270** | 1,65 | −5 863 | −1 656 | 41 / 572 |

- Sans limite (le Combine 5 lots ne contraint jamais le book), la validation fait 30 886 USD.
- À 2 lots fixes, la validation fait 25 857 USD, avec 279 séances sur 572 réduites.
- Sur le cycle Combine → XFA → payout, l'IS donne un cash net espéré de 607 USD, mais un cash **médian négatif** (−50 USD). La probabilité d'au moins un payout est de 79 %, celle de perdre l'XFA de 65 %.

### Verdict du holdout

- **Aucune donnée vierge n'existe sur disque** : tout s'arrête au 2026-10-02, et ces données ont déjà été lues.
- **(O) S01 ATR 18, du 2026-03-23 au 2026-10-02, 137 séances, partiellement contaminé : COHÉRENT.** Le PnL est de −390 USD pour un seuil pré-enregistré de −617,50 USD. Le gate technique (701 trades sur 701 reproduits) est réussi. Il n'y a eu que 11 trades.
- **(P), le suivi prospectif après le 2026-11-03, n'est pas lu.** Première lecture possible (H = 20) vers le 2026-12-03, après téléchargement des données.

### Règles d'arrêt proposées (à décider)

*Proposition corrigée le 2026-11-15 (écart n° 12).*

- **Alerte « revue »** : R3 (le PnL sur 120 séances passe sous le p5) ou R0 (la règle déjà enregistrée).
- **Passage en shadow** : R2 seule, quand le drawdown depuis le début du suivi dépasse le p99.
- **Réintégration** : après 252 séances de shadow, par décision versionnée.
- **Compromis assumé** :
  - l'action automatique coupe à tort environ 7 % des sleeves saines en 250 séances ;
  - en contrepartie, une dégradation n'est détectée que lentement : 14 à 23 % des sleeves à Sharpe vrai ≤ 0 en 250 séances ;
  - garder R0 comme déclencheur du shadow (« R0 ou R2 ») ferait monter les coupures à tort à 21 %.

---

## 2. Décisions qui vous reviennent

*État au 2026-11-15 :*
- *décision 6 prise : k = 3 ;*
- *décision 1 : l'option (b) a été testée et fermée (`CLOSED_IS`). Le 2026-11-15, l'utilisateur a **mis S01 de côté, à reprendre plus tard**. En attendant, S01 ATR 18 reste tel quel dans le lock du 15/11 ;*
- *décision 2 prise : R2 seule fait passer en shadow ; R0 et R3 sont des alertes « revue ». Règle versionnée [`stop_rule_20261115.json`](../strategy_registry/forward_monitoring/stop_rule_20261115.json) ;*
- *décision 3 prise : application rétroactive. S05, S07, S08 et S09 passent `ACTIVE_UNDER_REVIEW`, S04 reste en shadow, le book ne change pas ;*
- *décision 5 autorisée : téléchargement des données forward, plafonné à 5 USD, après la clôture du 2026-12-02 ;*
- *décision 4 : téléchargement des vendredis d'échéance MNQ autorisé, diagnostic en cours ;*
- *décision 6 (k = 3) : formalisée ; décision 7 (documentation) : faite, sauf l'export PDF des fiches ; décision 8 : la règle actuelle est conservée par défaut.*

1. **S01 s'assèche.**
   - *Mécanisme* :
     - Le filtre de S01 est l'ATR des bougies 1 minute au moment du signal. Un breakout n'est pris que si cet ATR tombe entre deux bornes figées en **points MNQ**, calibrées sur l'IS 2019-05 → 2024-02. Pour les 6 modèles ATR 18, ces bornes valent en médiane 7,75 et 20,07 points.
     - Une bougie d'une minute bouge en points à peu près proportionnellement au prix. MNQ est passé d'environ 12 000 en 2023 à 29 500 en 2026 : à agitation relative égale, l'ATR en points a plus que doublé.
   - *Diagnostic descriptif* : [diag_s01_drying.py](diag_s01_drying.py), [outputs/s01_drying_by_quarter.csv](outputs/s01_drying_by_quarter.csv). Aucun PnL n'est recalculé.

     | Trimestre | Prix MNQ | ATR 18 (points) | ATR / prix (pb) | Candidats au-dessus de la borne haute | Signaux retenus par le vote |
     |---|---:|---:|---:|---:|---:|
     | 2023 T2–T4 | 13 500–15 700 | 11–14 | 8,4–9,1 | 0–8 % | 38–42 |
     | 2022 T1–T2 | 12 600–14 600 | 24–25 | 16,7–18,7 | 72–76 % | 10–13 |
     | 2026 T1 | 24 950 | 30,8 | 12,4 | 90 % | 4 |
     | 2026 T2–T3 | 29 100–29 500 | 26–27 | 9,2–9,4 | 71–80 % | 11–15 |

   - *Lecture* :
     - En 2022, la borne haute coupait des journées réellement agitées : l'ATR rapporté au prix était de 17 à 19 pb, et le filtre jouait son rôle.
     - En 2026 T2–T3, l'ATR rapporté au prix (9 pb) est au niveau de 2023, et la borne haute coupe pourtant 71 à 80 % des candidats. Le filtre ne mesure plus l'agitation : il mesure le niveau de prix.
     - Le même biais existait en sens inverse en 2019, où la borne basse coupait 67 à 92 % des candidats.
   - *Options* :
     - (a) ne rien changer et laisser le forward juger : la sleeve contribuera peu.
     - (b) **Recommandé.** Ouvrir une nouvelle hypothèse pré-enregistrée avec des **bandes en ATR / prix**, quantiles calibrés sur l'IS de la fiche. Le sens économique du filtre ne change pas ; seul l'artefact d'échelle disparaît.
     - (b′) Variante secondaire : bornes = quantiles **roulants** de l'ATR en points, par exemple sur les 252 dernières séances. Elle s'adapte toute seule, mais retient par construction toujours la même part de jours, y compris dans un régime durablement agité comme 2022 : le filtre change alors de sens (agitation relative au passé récent). Elle ajoute aussi un paramètre de fenêtre.
     - (c) mettre S01 en revue.
   - *Cadre de (b)* :
     - sélection sur l'IS 2019-05 → 2024-02 uniquement ;
     - la validation 2024-02 → 2026-03 a déjà été vue, et la fenêtre 2026-03 → 10 a été consommée par (O) : le seul juge propre est le forward ;
     - S01 ATR 18 reste la référence tant que (b) n'a pas été validé ;
     - (b) n'a **pas** été testé ici : le faire dans l'audit serait un retuning.
   - **Suite du 2026-11-15 : (b) lancée hors audit, campagne fermée `CLOSED_IS`.** Voir [research/s01_atr_price_normalized_20261115](../research/s01_atr_price_normalized_20261115/HYPOTHESIS_CLOSURE.md).
     - **Bras A, bandes en ATR / prix** : il ne rend qu'une partie de la fréquence. Il fait 9, 15 et 8 trades sur 2026 T1–T3, contre 2, 9 et 2 pour la référence, soit un ratio de 0,52 pour un seuil pré-enregistré de 0,60. Son Sharpe IS est de 0,98, contre 1,25 pour la référence.
     - **Seconde cause de l'assèchement : le dimensionnement.** Un trade est sauté quand un seul contrat risque plus de 250 USD ; en 2026, cela arrive dès qu'il y a 0,4 % du prix entre l'entrée et le stop. Le bras B (plancher d'un contrat jusqu'à 500 USD de risque) retrouve la fréquence, mais son Sharpe IS tombe à 0,84 pour un seuil de 0,94. Les trades ajoutés ont perdu 935 USD sur l'IS.
     - **Aucun holdout lu** : NQ 2010–2019 reste vierge pour l'ORB. S01 ATR 18 est inchangé, et il reste à choisir entre (a) et (c).
2. **Règle d'arrêt.**
   - *Options* :
     - garder R0 seule : 17 % de fausses alarmes, 31 % de détection à Sharpe 0, délai d'environ 200 séances ;
     - **recommandé** : R3 et R0 en alerte, R2 seule pour le shadow ; 7 % de coupures à tort, 14 % de détection à Sharpe 0 ;
     - « R0 ou R2 » pour le shadow : 21 % de coupures à tort, 36 % de détection à Sharpe 0 ;
     - adopter R1 : plus rapide (environ 80 séances), mais 28 % de fausses alarmes.
   - Détail : [05_regles_arret.md](05_regles_arret.md).
3. **Statuts de S05, S07, S08 et S09.**
   - Appliquée à la lettre avec les données du 2026-11-01, la règle R0 en vigueur mettrait **S05 en shadow** : elle a franchi son seuil du 28/07 au 25/08/2026, puis est remontée. Elle n'a pas été appliquée, parce qu'elle n'était enregistrée que pour le forward.
   - **Avec la proposition corrigée, aucune sleeve active ne passe en shadow** : aucune n'a déclenché R2. S05, S07, S08 et S09 passent « en revue », c'est-à-dire en surveillance renforcée, sans que le book change. S04 reste en shadow, puisque R2 s'est déclenchée pour elle du 09/09 au 02/10. Voir [06_coherence_statuts.md](06_coherence_statuts.md).
   - *Options* : appliquer la règle rétroactivement à la fenêtre du replay, ou la garder strictement prospective. Dans ce cas, l'asymétrie avec S04, sortie sur cette même fenêtre, doit être assumée.
4. **Implémentation en production.**
   - La règle de netting r2 suppose que le moteur d'exécution calcule S01 avec l'ensemble ATR 18 et les seuils figés de `thresholds_is.csv`. Je n'ai rien changé côté exécution.
   - Il faut aussi trancher le **trou de données MNQ des vendredis d'échéance** : faut-il acquérir des données par contrat pour backtester ces jours-là, que la production tradera ?
5. **Téléchargement des données forward** (Databento, payant, avec garde-fou de coût), pour lire (P) à H = 20 vers le 2026-12-03, puis à 60, 120 et 250 séances.
6. **Stop catastrophe — DÉCIDÉ le 2026-11-15 : k = 3.**
   - Le stop vaut 3 × l'ATR journalier de J−1, posé chez le broker sur la position nette. Il ne s'est jamais déclenché sur l'historique. C'est une protection contre une panne ou un gap, pas un outil de rendement.
   - La borne journalière vient de la DLL Topstep de 1 000 USD, active par défaut. Aucun k testé ne borne une journée sous 500 USD.
   - **Formalisé le 15/11** dans la règle versionnée [`catastrophe_stop_20261115.json`](../strategy_registry/execution/catastrophe_stop_20261115.json), pointée par le registre, documentée dans `ACTIVE_EIGHT_SLEEVES.md` et vérifiée par `tests/test_catastrophe_stop_rule.py`. Reste à la coder dans votre plateforme d'exécution : le repo ne contient pas de moteur live.
7. **Documentation à corriger** :
   - BOOK.md, sections 1 à 5, calculées avec l'ancien S01 et sur le calendrier MNQ ;
   - la convention de Sharpe de la fiche 02 ;
   - le Sharpe de la fiche 01, calculé avec les dimanches ;
   - l'exemple du 31/01 dans BOOK.md et dans la fiche 01, déjà annoté.
8. **Exécution.** Aucune variante ne bat la règle actuelle sur l'IS : ni « pas de rattrapage tardif », ni priorité par contribution au risque, ni « pas de MBT sous 3 lots ». Je recommande de ne rien changer.

---

## 3. Bugs et écarts trouvés

| # | Écart | Gravité | État |
|---:|---|---|---|
| 1 | Le S01 des locks du 01/11 et du 03/11, du netting et du replay est `baseline_fixed` (823 trades), pas l'ensemble ATR 18 de la fiche (701 trades) | haute | **corrigé** par le lock du 15/11 et la règle de netting r2 |
| 2 | Le calendrier du PnL quotidien suit MNQ et perd 8 vendredis d'échéance tradés par MGC et MBT (−509,26 USD) | moyenne | **corrigé** par le lock du 15/11 |
| 3 | Le fichier MNQ `c.0` n'a aucune barre sur les 27 vendredis d'échéance de 2019–2025 (ni sur le 2026-03-20) : les sleeves MNQ n'ont jamais été testées ces jours-là | moyenne | ouvert |
| 4 | Le Sharpe publié de S02 (0,873) est calculé sur les rendements d'un compte de 50K qui capitalise ; en USD, il vaut 0,848 | faible | ouvert (doc) |
| 5 | Le Sharpe publié de S01 (1,156) est calculé sur un calendrier contenant 355 dimanches ; sur le calendrier MNQ, il vaut 1,281 | faible | ouvert (doc) |
| 6 | Le `summary.json` de S02 cite un hash périmé du manifeste ; le lock cite le bon | faible | ouvert |
| 7 | L'exemple 2 de BOOK.md (31/01/2024) repose sur un trade S01 qui n'existe pas dans la référence | faible | annoté |
| 8 | Le seuil R0 est le minimum de l'historique de la sleeve : il ne peut pas se déclencher sur les MBT, dont l'historique va jusqu'au 2026-10-02, alors qu'il peut se déclencher pour MNQ et MGC | moyenne | ouvert (décision 3) |
| 9 | Les bandes ATR de S01 sont en points absolus, d'où l'assèchement en 2026 : en 2026 T2–T3, 71 à 80 % des candidats dépassent la borne haute, pour un ATR rapporté au prix égal à celui de 2023. Le dimensionnement (trade sauté au-delà de 250 USD de risque par contrat) est une seconde cause, la plus forte en 2026 T3 : 11 signaux, 2 trades | haute | correctif ATR / prix testé : `CLOSED_IS` ; ouvert (décision 1) |
| 10 | Le vote S02 peut ouvrir une position jusqu'à 16:02, soit 6 minutes avant le flatten | faible | constat |
| 11 | La mission supposait l'absence de règle d'arrêt ; R0 existe depuis le 03/11 | — | pris en compte |
| 12 | **Erreur de l'audit lui-même** : la première version de la proposition annonçait environ 7 % de coupures à tort pour un shadow déclenché par « R2 ou R0 ». Ce chiffre ne valait que pour R2 seule ; la règle combinée, calculée le 15/11 sur les mêmes tirages, en donne 21 % | moyenne | **corrigé** : phases 5 et 6 relancées, proposition révisée (R0 devient une alerte) |

---

## 4. Phases

| Phase | Résultat principal | Livrable | Scripts |
|---|---|---|---|
| 0 — Reproduction | 85 contrôles sur 87 ; les 10 ledgers ré-exécutés depuis les barres brutes sont identiques ; deux écarts de définition tranchés (S01, Sharpe) | [00_reproduction.md](00_reproduction.md) | [phase0_reproduction.py](phase0_reproduction.py), [phase0_engine.py](phase0_engine.py) |
| 1 — Calendrier | 572 séances ; book à 8 sleeves : 30 858 USD, Sharpe 1,712, maxDD −5 870 ; book à 9 : 36 241, 1,509, −6 595 | [01_calendrier.md](01_calendrier.md) | [phase1_calendar.py](phase1_calendar.py) |
| 2 — Significativité | seule S01 est significative sur chaque fenêtre (DSR 0,91–0,97) ; S03 à S08 sont indistinguables ; le book est significatif (t de 2,8 à 4,8), par diversification | [02_significativite.md](02_significativite.md) | [phase2_significance.py](phase2_significance.py), [stats.py](stats.py) |
| 3 — Inventaire | pas de holdout vierge : tout s'arrête au 2026-10-02, déjà lu | [03_inventaire_holdout.md](03_inventaire_holdout.md) | [phase3_inventory.py](phase3_inventory.py) |
| 4 — Holdout | pré-enregistrement validé (`ca3b9b25…`) ; (O) COHÉRENT ; (P) à venir | [preregistration.md](preregistration.md), [04_holdout.md](04_holdout.md) | [phase4a_prereg.py](phase4a_prereg.py), [phase4a_validate.py](phase4a_validate.py), [phase4b_s01_oneshot.py](phase4b_s01_oneshot.py), [forward_check.py](forward_check.py) |
| 5 — Règles d'arrêt | aucune règle n'est à la fois rapide et sûre ; proposition à deux niveaux, corrigée le 15/11 (R2 seule → shadow ; « R0 ou R2 » donnerait 21 % de coupures à tort) | [05_regles_arret.md](05_regles_arret.md) | [phase5_stop_rules.py](phase5_stop_rules.py) |
| 6 — Cohérence des statuts | avec la proposition corrigée, S05, S07, S08 et S09 passent en revue et aucune sleeve active ne passe en shadow (S05 passerait en shadow avec « R0 ou R2 ») ; le LOO de S04 a été calculé sur la fenêtre réutilisée | [06_coherence_statuts.md](06_coherence_statuts.md) | [phase6_status.py](phase6_status.py) |
| 7 — Vote (exploratoire) | sur l'IS, le vote bat la moyenne de ses cellules pour 7 sleeves sur 8 ; la taille proportionnelle au score est rejetée ; le vote nul de S09 en OOS vient d'abord de ses coûts | [07_vote.md](07_vote.md) | [phase7_vote.py](phase7_vote.py), [phase7_s02_cells.py](phase7_s02_cells.py) |
| 8 — Régimes | le book perd 52 k USD les jours de range MNQ, via S01, S02 et S04 ; pas de pertes groupées à l'échelle du mois ; S02 porte 41 à 44 % du risque | [08_regimes.md](08_regimes.md) | [phase8_regimes.py](phase8_regimes.py) |
| 9 — Topstep | chiffres de tête sous contrainte ci-dessus ; la règle actuelle (i) est gardée | [09_topstep.md](09_topstep.md) | [phase9_topstep.py](phase9_topstep.py), [sim_variants.py](sim_variants.py) |
| 10 — Stop catastrophe | k = 3 (aucun déclenchement), **retenu le 2026-11-15** ; borne journalière par la DLL de 1 000 USD | [10_stop_catastrophe.md](10_stop_catastrophe.md) | [phase10_cat_stop.py](phase10_cat_stop.py) |
| Diagnostic S01 (hors `run_all`) | les bornes d'ATR en points coupent 71 à 80 % des candidats en 2026, à agitation relative égale à 2023 | [outputs/s01_drying_by_quarter.csv](outputs/s01_drying_by_quarter.csv) | [diag_s01_drying.py](diag_s01_drying.py) |

Les tables brutes sont dans [outputs/](outputs/), et la version HTML de chaque page dans [html/](html/RAPPORT.html).

---

## 5. Ce que l'audit ne prouve pas

- **Aucune performance hors échantillon propre.** Toutes les fenêtres de validation avaient été consultées avant le gel des clusters ; (O) est partiellement contaminé. Seul le forward après le 2026-11-03 sera un vrai test, et il faudra 120 à 250 séances pour qu'il discrimine.
- **Les distributions bootstrap supposent que le passé se répète.** Les seuils (pré-enregistrement, règles d'arrêt) sont estimés sur l'historique déjà vu, y compris les fenêtres réutilisées pour la sélection. Pour S01, dont la fréquence de trades a chuté, la distribution historique n'est plus représentative.
- **Les coûts sont nominaux.** Le slippage réel, les gaps à travers les stops, les coupures de flux et le comportement réel des ordres chez le broker ne sont pas testés.
- **Les données ont des trous** : vendredis d'échéance MNQ absents ; contrats continus `c.0` (MNQ) et `v.0` (MGC, MBT), non ajustés.
- **Les règles Topstep changent souvent.** 1 MBT = 1 lot reste à confirmer dans TopstepX. La DLL de 1 000 USD a été lue le 2026-11-15.
- **Le book n'est pas déflaté.** Le nombre de configurations de book essayées (choix des sleeves, sortie de S04) n'est pas documenté : son Sharpe de 1,7 à 1,9 est donc un plafond.
- **Les analyses de régime sont descriptives.** Le type de journée n'est connu qu'à la clôture ; rien n'en fait un filtre exploitable.
- **La phase 7 et les variantes de la phase 9 sont exploratoires.** Elles ne valident aucune modification.
- **Écart au cadre de la mission** : le lock du 15/11, la règle r2, le registre, la doc active et les tests ont été modifiés hors de `audit_20261115/`, sur votre instruction explicite du 15/11. Les références antérieures ne sont pas modifiées ; seuls une note `SUPERSEDED.md` et le test de registre de l'ancien lock ont été touchés.
