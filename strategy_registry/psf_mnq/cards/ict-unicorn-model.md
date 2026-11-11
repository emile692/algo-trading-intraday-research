# ICT Unicorn Model

- **ID** : `ict-unicorn-model`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `smart-money`
- **Famille de recherche (provisoire)** : `price_action_reversal`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/ict-unicorn-model) · [page MNQ](https://pinescriptforge.com/mnq/ict-unicorn-model/backtest)

## Description publiée

Combines Fair Value Gap and Order Block for the highest-conviction ICT entry model.

- **Entrée annoncée** : Identify an Order Block that contains a Fair Value Gap within its range. Enter when price retraces into the overlapping zone with a bullish/bearish confirmation candle.
- **Sortie annoncée** : Target the recent swing high/low or opposing liquidity pool. Stop below/above the Order Block.

## Données et formalisation

- **Indicateurs cités** : Order Block, Fair Value Gap, Market Structure
- **Horizons cités** : 5min, 15min, 1H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
