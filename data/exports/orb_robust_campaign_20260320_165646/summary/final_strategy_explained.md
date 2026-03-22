# Stratégie Finale - Explication

## Que signifie `b_atr_q20_q90`

Le nom se lit comme suit:
- `b`: bloc de recherche **B** (régime de volatilité ATR).
- `atr`: la feature filtrante est l'**ATR(14)** au moment du signal.
- `q20_q90`: on garde uniquement les signaux dont l'ATR est entre le **20e percentile** et le **90e percentile** de la distribution ATR observée pendant la phase diagnostic.

Dans ce run, les seuils numériques sont:
- ATR min = **6.8393**
- ATR max = **21.5464**

Autrement dit, on évite:
- les régimes de volatilité trop faibles (sous q20),
- les régimes trop extrêmes (au-dessus de q90).

## Règles complètes de la stratégie finale

La logique ORB reste la baseline, avec un seul filtre additionnel ATR.

Paramètres de base conservés:
- Instrument/dataset: `MNQ_1mim.parquet`
- ORB long only
- `or_minutes = 15`
- `opening_time = 09:30:00`
- `one_trade_per_day = True`
- `entry_buffer_ticks = 2`
- `stop_buffer_ticks = 2`
- `target_multiple = 2.0`
- `vwap_confirmation = True`
- `vwap_column = continuous_session_vwap`
- `time_exit = 16:00:00`
- `account_size_usd = 50000`
- `risk_per_trade_pct = 0.5`

Filtre ajouté:
- Accepter le trade seulement si `ATR(14)` du signal est dans `[q20, q90]`, ici `[6.8393, 21.5464]`.

## Pourquoi ce filtre est retenu

Le filtre améliore la robustesse sans complexifier la stratégie.

Résultats globaux (baseline vs final):
- Trades: `1125` -> `788`
- Profit factor: `1.1477` -> `1.2945`
- Sharpe: `0.6899` -> `1.0893`
- Expectancy: `13.3947` -> `25.4708`
- Max drawdown: `-2659.5` -> `-2083.5`
- Cumulative PnL: `15069.0` -> `20071.0`

Résultats out-of-sample (30% final):
- Trades: `288` -> `222`
- Profit factor: `1.1420` -> `1.2421`
- Sharpe: `0.6129` -> `0.8796`
- Expectancy: `13.0208` -> `21.2140`
- Max drawdown: `-1879.0` -> `-1397.5`
- Cumulative PnL: `3750.0` -> `4709.5`

## Point d'attention

La stratégie finale trade moins souvent que la baseline.

Conséquence:
- meilleure qualité moyenne des trades,
- mais plus faible participation.

Il faut donc valider que cette baisse de fréquence est acceptable pour ton objectif opérationnel.

## Références d'artefacts

- Campagne finale: `data/exports/orb_robust_campaign_20260320_165646/`
- Rapport final: `data/exports/orb_robust_campaign_20260320_165646/summary/final_campaign_report.md`
- Résumé de campagne: `data/exports/orb_robust_campaign_20260320_165646/summary/campaign_summary.md`
- Tableau variantes: `data/exports/orb_robust_campaign_20260320_165646/campaigns/variants_consolidated.csv`
