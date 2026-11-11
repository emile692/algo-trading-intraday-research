# Analyse individuelle des neuf sleeves gelées

## Verdict

**TRADEOFF — aucune modification du book actif.** Les variantes sont un diagnostic de capacité et de compatibilité prop, pas une sélection a posteriori.

## Points d'audit déterminants

- S01 était réellement sizée à **250 USD de risque nominal** avec quantité entière variable dans son ledger source. La référence native de cette campagne est sa reconstruction linéaire à un micro.
- S02 exécute un **time-stop de 240 minutes**. Le niveau RTH open n'est pas un stop exécuté et le sizing USD est donc non interprétable.
- S07–S09 sont les stratégies exécutables agrégées par vote, une position MBT maximum chacune ; les cellules sous-jacentes ne sont pas comptées comme ordres.
- Sizing USD non interprétable, quantités fixes seulement : S02, S07, S08, S09.
- Les périodes postérieures aux anciens splits ont déjà été inspectées pendant la sélection : elles mesurent la stabilité, pas un nouvel OOS indépendant.

## Référence native (un micro, coûts nominaux)

| sleeve | signals | net_pnl_usd | sharpe_zero_days | profit_factor | max_drawdown_usd | worst_day_usd | worst_trade_usd | mean_holding_minutes_executed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mnq_orb_vwap_atr | 823 | 14130.0 | 1.068 | 1.252 | -1732.5 | -249.5 | -249.5 | 203.423 |
| mnq_volatility_breakout | 899 | 18597.0 | 0.897 | 1.285 | -4655.5 | -925.0 | -925.0 | 197.461 |
| mnq_compression_breakout | 331 | 2617.5 | 0.489 | 1.227 | -1190.0 | -236.5 | -236.5 | 81.462 |
| mnq_atr_trend_flip | 2119 | 12377.5 | 0.509 | 1.094 | -4056.0 | -781.0 | -359.5 | 127.419 |
| mgc_volatility_breakout | 199 | 3240.5 | 0.384 | 1.34 | -1481.5 | -691.5 | -691.5 | 168.367 |
| mgc_gap_continuation | 236 | 7604.0 | 0.733 | 1.545 | -1320.0 | -883.5 | -883.5 | 332.059 |
| mbt_rolling_range_breakout | 288 | 1993.34 | 0.673 | 1.193 | -1285.8 | -355.32 | -355.32 | 278.25 |
| mbt_trend_pullback_ma_structure | 693 | 3465.24 | 0.827 | 1.153 | -1653.28 | -533.64 | -338.32 | 197.167 |
| mbt_volatility_breakout | 442 | 2244.06 | 0.565 | 1.12 | -1916.4 | -229.32 | -229.32 | 221.618 |

## Lecture centrale

La granularité domine aux petits budgets : à 100 USD, S01 conserve 111/823 signaux, S03 245/331, S04 1214/2119, S05 0/199 et S06 56/236. S05 est donc entièrement éteinte à 100 USD. À 250 USD, la rétention devient 822/823, 327/331, 2091/2119, 99/199 et 189/236 respectivement. Le contrôle `filter_q1` sépare cette destruction de fréquence de la simple mise à l'échelle.
En standalone 50K à 60 séances et q1, les PASS restent faibles (0–5,27 %) ; S02 est la plus rapide mais aussi sensiblement plus exposée au MLL (9,6 %), tandis que S04 atteint 4,07 % de PASS avec 16,2 % de MLL. La hausse de quantité accélère donc la progression au prix d'un drawdown et d'un risque MLL nettement supérieurs. Cela ne constitue pas une amélioration robuste générale.

## Topstep — règles vérifiées le 2026-10-13

Comptes 50K/100K/150K : objectifs 3K/6K/9K, MLL 2K/3K/4,5K, limites 50/100/150 micros et règle de cohérence 55 %. Le MLL suit le solde EOD, est contrôlé en temps réel avec PnL latent et se verrouille au solde initial. Les fins de simulation sont `UNRESOLVED`, jamais `EXPIRY`.

## Fichiers

Les métriques, distributions de quantité, audit du risque prévu versus fill et perte réalisée, années, sous-périodes, intervalles block-bootstrap et simulations Topstep appariées sont fournis dans les CSV voisins. Les hypothèses reproductibles sont dans `config.json` et `PROTOCOL.md`; les sources et conventions de chaque sleeve sont dans `IMPLEMENTATION_AUDIT.md`.
