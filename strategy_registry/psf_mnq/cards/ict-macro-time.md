# ICT Macro Time

- **ID** : `ict-macro-time`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `smart-money`
- **Famille de recherche (provisoire)** : `price_action_reversal`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/ict-macro-time) · [page MNQ](https://pinescriptforge.com/mnq/ict-macro-time/backtest)

## Description publiée

Trades specific ICT macro time windows where algorithmic activity creates opportunities.

- **Entrée annoncée** : Only take entries during macro time windows. Look for displacement + FVG + order block confluence.
- **Sortie annoncée** : Target the next liquidity pool above/below. Stop beyond the macro time candle extreme.

## Données et formalisation

- **Indicateurs cités** : Macro Time Windows, FVG, Order Blocks
- **Horizons cités** : 1min, 5min, 15min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
