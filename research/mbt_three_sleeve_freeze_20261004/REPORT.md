# MBT — freeze de trois régions et diagnostic portefeuille

Date de décision : 4 octobre 2026.

## Décision enregistrée

Les trois régions candidates sont gelées, sans sélection ultérieure d'une meilleure cellule :

| Sleeve | Bornes gelées | Cellules déjà testées utilisées dans le diagnostic |
|---|---|---|
| `rolling_range_breakout` | `entry_channel_length` 20–24; `exit_channel_length` 7–15 | 2 × 4 = 8 |
| `trend_pullback_ma_structure` | `fast_pair_scale` 0,50–0,75; `slow_tail_scale` 0,75–1,25 | 3 × 3 = 9 |
| `volatility_breakout` | `entry_atr_multiple` 0,30–0,60; `daily_atr_period` 10–30 | 3 × 4 = 12 |

Pour ne pas choisir la meilleure cellule après observation, chaque sleeve de recherche est la moyenne équipondérée du PnL quotidien des cellules testées dans sa région. Ce composite fractionnaire sert au diagnostic de portefeuille; ce n'est pas encore une règle exécutable à un contrat MBT. Une spécification de production devra fixer à l'avance une cellule représentative ou une règle d'agrégation de signaux.

`momentum_confirmed_breakout` n'est ni gelé ni inclus. Sa région candidate (`ADX` 15–20, `lookback` 20–40) reste conditionnée à une validation sur données indépendantes.

## Sharpe des trois nouvelles sleeves

Même moteur, mêmes coûts et même calendrier que la campagne MBT : 1 MBT, commission round-turn 2,82 USD et un tick de slippage par côté.

| Série | Historique complet 2021-05-03–2026-10-02 | Ancien IS | Ancien OOS 2025-05-29–2026-10-02 |
|---|---:|---:|---:|
| Rolling range | 0,424 | 0,258 | 0,755 |
| Trend pullback MA | 0,573 | 0,490 | 0,742 |
| Volatility breakout | 1,066 | 1,321 | 0,559 |
| **Panier 3 sleeves, equal-weight brut** | **0,895** | **0,932** | **0,842** |
| Panier 3 sleeves, equal-risk diagnostique | 0,895 | 0,928 | 0,845 |

Le panier brut complet produit 3 495,54 USD nets, avec un max drawdown de -1 011,38 USD. Les cellules de chaque région restent toutes positives sur l'ancien OOS : Sharpe 0,720–0,798 pour rolling range, 0,473–1,068 pour trend pullback, et 0,236–0,834 pour volatility breakout.

Attention : cet « ancien OOS » a servi à lire les heatmaps et à définir les régions. Il n'est donc plus indépendant et ne constitue pas une confirmation OOS après freeze.

## Intégration avec les six sleeves actuelles

La comparaison strictement alignée utilise 505 séances communes du 3 janvier 2024 au 19 mars 2026. Le Sharpe des six sleeves est donc recalculé sur ces mêmes dates et diffère du chiffre canonique sur 564 séances.

| Allocation | 6 actuelles | 3 MBT | 9 sleeves | Gain 6→9 |
|---|---:|---:|---:|---:|
| Equal-weight brut | 1,513 | 1,425 | **1,800** | **+0,287** |
| Equal-risk diagnostique | 1,464 | 1,413 | **1,902** | **+0,438** |

Sur la portion commune de l'ancien OOS (190 séances, 29 mai 2025–19 mars 2026), les Sharpes equal-weight bruts sont respectivement 1,337, 1,734 et 1,764; le gain 6→9 est +0,427. En equal-risk, ils sont 1,322, 1,717 et 1,969.

Le gain est encourageant mais incertain : bootstrap mobile par blocs de 10 séances, 5 000 réplications, IC 95 % du delta de Sharpe equal-weight = [-0,093; +0,802] sur toute la fenêtre commune et [-0,476; +1,371] sur sa portion ancien OOS. Les deux intervalles incluent zéro.

## Orthogonalité

| Diagnostic | Fenêtre commune | Portion ancien OOS commune |
|---|---:|---:|
| Corrélation panier 3 MBT vs livre 6 | **0,197** | **0,281** |
| Corrélation rolling range vs livre 6 | 0,080 | 0,113 |
| Corrélation trend pullback vs livre 6 | 0,114 | 0,187 |
| Corrélation volatility breakout vs livre 6 | 0,260 | 0,383 |
| Corrélation absolue moyenne, 18 couples 3×6 | 0,070 | 0,113 |
| Corrélation absolue maximale, 18 couples 3×6 | 0,282 | 0,414 |

Le bootstrap donne un IC 95 % de [0,105; 0,313] pour la corrélation des deux livres sur la fenêtre commune, et [0,051; 0,471] sur la portion ancien OOS. L'orthogonalité est donc bonne mais pas totale, et se dégrade récemment.

Les régressions linéaires des nouvelles sleeves sur les six actuelles expliquent seulement 1,5 % de la variance de rolling range, 5,3 % de trend pullback et 9,6 % de volatility breakout. Le nombre effectif de dimensions de risque passe de 4,69 pour les six actuelles à 6,64 pour les neuf; les trois MBT seules valent 2,18 dimensions.

Entre elles, les trois nouvelles sleeves ne sont pas indépendantes : corrélations de 0,224 (rolling/volatility), 0,306 (trend/volatility) et 0,629 (rolling/trend). Le principal doublon externe est logiquement `volatility_breakout`, corrélé à 0,282 avec la sleeve MNQ volatility sur toute la fenêtre, et à 0,414 avec MNQ ATR trend sur la portion récente.

Pendant les 10 % pires journées du livre actuel, le livre MBT perd en moyenne 20,85 USD contre -105,73 USD pour le livre actuel; sur la portion récente, -5,70 USD contre -109,31 USD. Il amortit donc fortement les mauvais jours, sans être un hedge systématiquement positif.

## Conclusion de gouvernance

Les trois régions peuvent être enregistrées comme **freeze candidates** et leur combinaison est économiquement prometteuse : Sharpe 0,90 sur l'historique MBT, faible corrélation moyenne avec les six sleeves, et amélioration ponctuelle du portefeuille 6→9. En revanche, ni le gain de Sharpe ni la persistance de l'alpha ne sont validés sur données indépendantes. La prochaine observation postérieure au 2 octobre 2026 doit être conservée intacte, sans retuning de ces bornes ni de la future règle d'exécution.

