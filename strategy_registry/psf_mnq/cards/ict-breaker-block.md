# ICT Breaker Block

- **ID** : `ict-breaker-block`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `smart-money`
- **Famille de recherche (provisoire)** : `price_action_reversal`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/ict-breaker-block) · [page MNQ](https://pinescriptforge.com/mnq/ict-breaker-block/backtest)

## Description publiée

Trades failed order blocks that convert into breaker blocks for high-probability reversals.

- **Entrée annoncée** : Identify a prior bearish order block that price broke through to the upside (failed OB). When price retraces back down to this zone, it becomes a bullish breaker block — enter long with a bullish rejection candle. Opposite for bearish breakers.
- **Sortie annoncée** : Target the high of the move that broke the original OB, or the next liquidity pool. Stop below the breaker block low. Minimum 2.5R target.

## Données et formalisation

- **Indicateurs cités** : Breaker Block Detection, Order Block Failure, Market Structure, Liquidity Pools
- **Horizons cités** : 5min, 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
