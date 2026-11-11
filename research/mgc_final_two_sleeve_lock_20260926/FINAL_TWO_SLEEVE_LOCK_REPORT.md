# MGC — verrouillage final de deux sleeves

## Statut méthodologique

- Configuration verrouillée avant ce calcul : `c47fafaac2c9cffc83bf5a7ef7452f26557d8bbc87fd053ab3244221436b5797`.
- La Validation 2024-01-01 → 2026-03-19 a déjà servi à cibler ces tests : elle est **adaptative et non OOS vierge**.
- Aucun nouveau paramètre ni nouvelle famille n'a été exploré après ce passage.
- Les représentants sont choisis par règle de plateau, jamais par maximum de Sharpe.

## Volatility breakout — trois représentants fermés

| Point | Sharpe IS | Sharpe Val | PF IS | PF Val | Trades IS | Trades Val | DD IS USD | DD Val USD | Années complètes + | Décision brute |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `volatility_10_0.8` | 0.322 | 0.534 | 1.118 | 1.283 | 297 | 124 | -1562 | -1906 | 4/6 | FREEZE |
| `volatility_30_1.1` | 0.431 | 0.485 | 1.266 | 1.404 | 134 | 65 | -1148 | -1481 | 4/6 | FREEZE |
| `volatility_50_1.3` | 0.487 | 0.472 | 1.421 | 1.487 | 79 | 42 | -940 | -1187 | 4/6 | FREEZE |

Représentant imposé par la règle centrale : `30 / 1.10`. Décision : **FREEZE CANDIDATE**. Il compte 4/6 années complètes positives; son voisinage fin contient 6/9 cellules positives dans les deux périodes, avec Sharpe conservateur voisin médian 0.21.

## Gap continuation — raffinement local fermé

- Cellules robustes (trades, Sharpe et PF) : `81/81`.
- Plus grande composante 4-connectée : `81` cellules.
- Médoïde géométrique : `minimum_gap_atr_multiple=0.5`, `volume_threshold_multiple=0.5`.
- Sharpe IS/Validation : `0.805/0.621`; PF `1.465/1.341`; trades `146/102`; DD `-939/-1403 USD`.
- Années complètes positives : `6/6`; voisinage robuste `9/9`.
- Décision : **FREEZE CANDIDATE**.

## Provenance des données

- Parquet MGC v0 imposé : `data\processed\parquet\MGC_v_0_1m_20150101_20260923.parquet`.
- SHA-256 vérifié : `2bbc2d10d59409fd347779fc00a23633fa6ad6aec0cf4b52e7a6d7a278426e96`.

## Freeze

- Candidats écrits au manifeste : `2`.
- SHA-256 du manifeste : `7a08230d42cc4e72fd6d7a408650cd64639970c60ee58a1de38dbe1f3a156628`.
- Les specs portent `parameter_optimization_after_lock=false` et remplacent uniquement le statut de candidat; aucun fichier baseline source ou ancien verrou n'est modifié.
- Toute évaluation future doit être prospective ou sur un segment réellement vierge, sans retuning.

## Amendement utilisateur — gap_continuation

- Représentant final remplacé par `minimum_gap_atr_multiple=0.50`, `volume_threshold_multiple=0.60`.
- Choix explicite à l'intérieur du plateau 81/81; aucune recherche ni comparaison supplémentaire.
- Contrôle exact : Sharpe IS/Validation `0.872/0.864`; PF `1.529/1.556`; trades `141/95`; DD `-968/-1320 USD`.
- Années complètes positives : `6/6`; gates de freeze : **PASS**.
- Nouveau SHA-256 du manifeste : `018eb4dd9bdb2896e2b4e759dc343a3ba8dce51f87038f2a252bcdb1efc9c1cf`.
