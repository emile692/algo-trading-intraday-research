# Sleeves MNQ rejouées sur `MNQ.v.0` au lieu de `MNQ.c.0` (2026-11-20)

**Diagnostic descriptif. Aucun paramètre n'est modifié, et rien ici ne doit servir à retoucher les sleeves.**

## Question

Les ledgers gelés des sleeves MNQ (S01–S04) ont été calculés sur `MNQ.c.0`. Cette série pose trois problèmes :
- **(a)** elle garde le contrat qui expire pendant la semaine du roll, quand son volume tombe à environ 25 % de la normale ;
- **(b)** elle n'a aucune barre RTH les vendredis d'échéance ;
- **(c)** elle fait apparaître un faux gap le jour du changement de contrat. Cet écart vaut 99,6 points en médiane sur les 28 rolls, et jusqu'à 276 points. Il entre dans l'ATR quotidien.

La production, elle, trade le contrat le plus échangé, c'est-à-dire ce que suit `v.0`.

## Méthode

Les moteurs de [diag_expiry_fridays.py](../../audit_20261115/diag_expiry_fridays.py) sont appelés sans aucune modification ; seul le fichier de barres change. Les ledgers gelés se reproduisent à l'identique sur `c.0` (audit, phase 0). Les deux variantes s'arrêtent à la même date que l'entrée gelée (2026-03-20 09:29 ET) :

| Variante | Problèmes supprimés | Problème restant |
|---|---|---|
| **V0_RAW** : `v.0` tel que téléchargé | (a) et (b) | le faux gap (c), qui tombe alors le jour du changement de contrat de `v.0` |
| **V0_ADJ** : `v.0` ajusté en arrière de l'écart de roll | (a), (b) et (c) | le niveau des prix anciens est décalé |

## Résultats

Fenêtres des ledgers gelés : du 2019-05-06 au 2026-03-19, sauf S01 (du 2019-05-05 au 2026-03-20).

| Sleeve | Trades gelés | Trades identiques (ADJ) | PnL gelé | PnL V0_ADJ | PnL V0_RAW |
|---|---|---|---|---|---|
| S01 orb_vwap_atr | 701 | 694 (99 %) | 19 417 | 18 617 (−800) | 18 617 |
| S02 volatility_breakout | 1 085 | 857 (79 %) | 20 259 | 22 747 (+2 488) | 22 007 (+1 748) |
| S03 compression_breakout | 329 | 324 (98 %) | 2 256 | 2 612 (+357) | 2 612 |
| S04 atr_trend_flip (shadow) | 2 052 | 2 007 (98 %) | 10 075 | 7 551 (**−2 525**) | 6 747 (**−3 328**) |

**Book actif S01 + S02 + S03.** Calcul sur les jours ouvrés, journées sans trade incluses :

| | PnL (USD) | Sharpe | Drawdown max (USD) |
|---|---|---|---|
| Gelé (`c.0`) | 41 931 | 1,34 | 5 027 |
| V0_ADJ | 43 976 | 1,42 | 4 655 |
| V0_RAW | 43 236 | 1,40 | 4 707 |

Sur la période après l'IS (2024-01 → 2026-03), le PnL du book actif vaut 15 436 USD sur la version gelée, 16 218 sur V0_ADJ et 15 352 sur V0_RAW.

## Lecture

1. **Le book actif n'est pas flatté par `c.0` ; il serait même légèrement meilleur sur `v.0`.** S01 et S03 ne dépendent pas du faux gap : leurs résultats V0_RAW et V0_ADJ sont identiques.
2. **S02 dépend fortement de la série, trade par trade.**
   - 21 % de ses trades changent.
   - 189 des 228 trades disparus tombent **hors** des semaines de roll. L'ATR de Wilder et le vote de cellules propagent un petit écart de données sur des semaines entières.
   - Le total reste stable, même un peu meilleur sur `v.0`.
   - Le ledger gelé de S02 ne doit donc pas servir de référence trade par trade pour le suivi en production : seuls ses agrégats sont robustes.
3. **S04 (shadow) était flatté par `c.0`, de −25 à −33 %.**
   - Les trades qui changent se trouvent presque tous dans les fenêtres de roll : 78 trades sur 82 ajoutés, pour −3 618 USD, et 43 sur 45 disparus, pour −1 834 USD.
   - Environ −630 USD viennent des vendredis d'échéance, déjà mesurés par l'audit.
   - À verser au dossier de la décision sur S04 et sa règle d'arrêt.

Fichiers : [replay.py](replay.py) · [outputs/replay_summary.csv](outputs/replay_summary.csv) · [outputs/v0_roll_spreads.csv](outputs/v0_roll_spreads.csv) · les ledgers rejoués sont dans `cache/`, non suivi par git.
