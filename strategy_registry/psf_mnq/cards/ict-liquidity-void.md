# ICT Liquidity Void

- **ID** : `ict-liquidity-void`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `smart-money`
- **Famille de recherche (provisoire)** : `price_action_reversal`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/ict-liquidity-void) · [page MNQ](https://pinescriptforge.com/mnq/ict-liquidity-void/backtest)

## Description publiée

Identifies and trades large price voids where minimal trading occurred — expecting price to return and fill.

- **Entrée annoncée** : Identify a Liquidity Void (large candle range with minimal wicks and volume gaps). When price retraces to the void, enter in the direction of the original displacement at the 50% level of the void.
- **Sortie annoncée** : Target the high/low of the void completion. Stop beyond the full void. Trail stop as the void fills progressively.

## Données et formalisation

- **Indicateurs cités** : Liquidity Void Detection, Volume Analysis, Displacement Candles
- **Horizons cités** : 5min, 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
