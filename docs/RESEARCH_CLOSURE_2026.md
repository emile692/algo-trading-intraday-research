# Clôture du programme de recherche cross-asset

Date de décision : **26 septembre 2026**.

## Décision générale

La recherche de nouveaux sleeves est arrêtée. Le portefeuille de référence contient quatre sleeves MNQ et deux sleeves MGC. Aucune branche MES/M2K, 6E, ZN ou MCL ne doit être rouverte sur les mêmes données par une nouvelle grille ou un ajustement de seuils.

L'objectif était un processus de PnL indépendant et robuste, pas le meilleur Sharpe ponctuel. Les branches ont été fermées lorsqu'elles n'ont pas simultanément démontré stabilité IS/OOS, taille d'échantillon, robustesse locale et résistance aux coûts.

## Synthèse

| Branche | Étendue testée | Résultat | Décision |
|---|---|---|---|
| MES | Réplication locale de 4 mécanismes MNQ | Quelques cellules positives, mais aucun plateau suffisamment robuste et cohérent; paramètres centraux faibles ou négatifs en validation | STOP |
| M2K | Réplication locale de 4 mécanismes MNQ | Pas de réplication convaincante; ORB, ATR trend et volatility breakout classés asset-specific | STOP |
| 6E | 16 mécanismes frozen, paysages 7×7, puis 3 frameworks FX spécifiques | Aucun plateau robuste des 16; les 3 frameworks ciblés sont tous négatifs OOS après coûts | STOP définitif |
| ZN | 16 mécanismes frozen, paysages 7×7, puis 3 frameworks rates spécifiques | Baselines très largement négatives; les 3 frameworks ciblés sont négatifs IS/OOS | STOP définitif |
| MCL | 16 mécanismes frozen, paysages 7×7, puis validation ciblée de 2 structures | Aucun sleeve ne résiste simultanément au temps, aux coûts et aux tests locaux | STOP définitif |

## MES et M2K

Fenêtres : IS 2019-05-06–2023-12-31, validation 2024-01-01–2026-03-19. Une cellule robuste exigeait Sharpe positif, PF > 1 et au moins 20 trades dans chaque période; un plateau devait contenir au moins trois cellules contiguës et s'étendre sur les deux axes.

### MES

- `atr_trend_flip` : aucun plateau commun; point central Sharpe IS/validation `-0,50 / 0,28`.
- `orb_vwap_atr` : point central `0,56 / -0,07`, PF validation `0,98`; la réplication du sleeve MNQ échoue.
- `volatility_breakout` : point central `0,10 / 0,15`, avantage trop faible; seulement une cellule dans la plus grande zone robuste.
- `compression_breakout` : `0,11 / 0,24`, mais seulement 97 trades validation et zone étroite; conclusion initiale `inconclusive`, jamais promue en sleeve.

### M2K

- `atr_trend_flip` : `-0,29 / -0,64`.
- `orb_vwap_atr` : `0,02 / -0,26`.
- `volatility_breakout` : `-0,82 / 0,03`.
- `compression_breakout` : `0,11 / -0,23` au point central; petit îlot positif ailleurs, insuffisant pour un lock.

Conclusion : absence de réplication structurelle suffisamment large. Les mécanismes retenus restent spécifiques au bloc MNQ dans cette étude.

## 6E

### Transposition des 16 mécanismes

776 cellules structurelles ont été exécutées sur deux périodes. La grande majorité des mécanismes n'a aucune cellule simultanément positive IS/OOS. Les seuls recouvrements sont de petits îlots :

- `range_structure_reversion` : 2/49 cellules communes, composante maximale 1;
- `regression_equilibrium_reversion` : 4/49, composante maximale 2;
- `short_horizon_pullback_reversion` : 2/49, composante maximale 1.

Aucune structure ne forme un plateau robuste avec fréquence crédible.

### Campagne ciblée FX

| Framework | Sharpe IS | Sharpe OOS | PnL OOS USD | PF OOS |
|---|---:|---:|---:|---:|
| `eur_benchmark_fix_reversal` | 0,10 | -0,39 | -6 126 | 0,95 |
| `regional_working_hours_fx_flow` | 0,11 | -0,59 | -9 189 | 0,93 |
| `scheduled_macro_acceptance` | -0,20 | -1,22 | -3 189 | 0,51 |

Les trois perturbations locales de chaque framework sont négatives OOS. Les deux règles de flux FX redeviennent positives à zéro slippage mais échouent avec le coût nominal : l'effet brut n'est pas exploitable de façon robuste.

## ZN

### Transposition des 16 mécanismes

Les 16 spécifications canoniques sont toutes négatives IS. En OOS, `statistical_zscore_reversion` est proche de zéro (Sharpe -0,06, PF 0,98); toutes les autres sont nettement négatives. Dans les paysages 7×7, seul `short_horizon_pullback_reversion` possède deux cellules communes contiguës; l'échantillon et le voisinage ne justifient pas un lock.

### Campagne ciblée rates

| Framework | Sharpe IS | Sharpe OOS | PnL OOS USD | PF OOS |
|---|---:|---:|---:|---:|
| `ten_year_auction_concession_reversal` | -1,12 | -0,11 | -299 | 0,94 |
| `scheduled_treasury_information_assimilation` | -0,72 | -0,84 | -1 824 | 0,68 |
| `treasury_day_night_segmentation` | -2,32 | -2,05 | -22 315 | 0,70 |

Le vrai post-échantillon des adjudications, à partir du 1er août 2024, reste négatif : PnL -494 USD, Sharpe -0,22, PF 0,89. La segmentation day/night est négative chaque année et son rolling Sharpe 252 jours médian vaut -2,01.

## MCL

### Transposition des 16 mécanismes

- Seul `atr_trend_flip` est positif IS/OOS, mais avec Sharpe ≈0,1 et PF ≈1,02 : marge économique insuffisante.
- `statistical_zscore_reversion` atteint 0,81 OOS mais est négatif IS; le retenir aurait été post-hoc.
- `gap_continuation`, `ma_crossover_trend` et `volatility_breakout` sont positifs IS puis négatifs OOS.
- Les autres mécanismes sont négatifs ou disposent d'un échantillon trop faible.

### Validation finale de deux structures

`short_horizon_pullback_reversion` : cinq points diagonaux ont seulement 1–11 trades IS et 1–5 OOS. La diagonale correspond à des événements RSI extrêmement rares (0,008% à 0,129% des barres éligibles), pas à un plateau statistiquement validable.

`ma_crossover_trend` : cinq représentants centraux sont positifs au coût nominal, mais les cinq perdent en 2024. Seulement 2/5 restent positifs OOS à deux ticks de slippage; 0/5 à trois ticks ou avec commission doublée.

Conclusion : les deux mécanismes sont `ABANDON_AS_MCL_SLEEVE`; les quatorze autres restent `FROZEN_NOT_REOPENED`.

## Autres pistes non retenues

- Variantes de sizing MNQ 3-state : écartées au profit du sizing nominal de l'ORB retenu.
- Nombreuses variantes ORB, VWAP, volume-climax et overlays prop-firm : exploratoires ou remplacées par les locks finals; elles ne font pas partie des six sleeves.
- MGC : seuls `volatility_breakout` et `gap_continuation` survivent. Les autres mécanismes et raffinements intermédiaires sont supersédés par le manifeste final.

## Enseignements méthodologiques

1. Une amélioration OOS isolée ne compense pas un IS négatif.
2. Un Sharpe élevé sur quelques trades n'est pas une preuve.
3. Une cellule positive doit appartenir à une structure locale stable.
4. Les coûts réalistes éliminent plusieurs effets bruts apparemment positifs.
5. La diversification nominale par actif ne suffit pas : le processus de PnL doit être économiquement distinct et persistant.
6. Les périodes déjà utilisées pour diagnostics ne doivent plus être qualifiées d'OOS vierges.

## État final

- Recherche de nouveaux sleeves : **CLOSED**.
- Branches MES/M2K, 6E, ZN, MCL : **DO NOT REOPEN ON SAME DATA**.
- Portefeuille suivi : **4 MNQ + 2 MGC**.
- Prochaine activité légitime : suivi prospectif, contrôle d'exécution et gouvernance du risque des six sleeves gelées.
