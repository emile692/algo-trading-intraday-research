# ICT Fair Value Gap (FVG)

- **ID** : `ict-fair-value-gap`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `smart-money`
- **Famille de recherche (provisoire)** : `price_action_reversal`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/ict-fair-value-gap) · [page MNQ](https://pinescriptforge.com/mnq/ict-fair-value-gap/backtest)

## Description publiée

Trades price returning to fill imbalance gaps left by institutional displacement.

- **Entrée annoncée** : Identify a 3-candle FVG in the direction of the higher timeframe trend. Enter long when price retraces into a bullish FVG (typically 50% of the gap). Enter short on bearish FVG retest.
- **Sortie annoncée** : Target the high/low of the candle that created the FVG or the next liquidity pool. Stop beyond the full FVG. Scale out at 50% and 100% of the gap.

## Données et formalisation

- **Indicateurs cités** : Fair Value Gap Detection, Higher Timeframe Bias, Structure
- **Horizons cités** : 1min, 5min, 15min, 1H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
