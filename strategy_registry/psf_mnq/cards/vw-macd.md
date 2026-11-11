# Volume-Weighted MACD

- **ID** : `vw-macd`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `volume`
- **Famille de recherche (provisoire)** : `volume_auction`
- **Rôle envisagé (provisoire)** : `alpha_or_feature`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/vw-macd) · [page MNQ](https://pinescriptforge.com/mnq/vw-macd/backtest)

## Description publiée

MACD calculated with volume-weighted moving averages for confirmation.

- **Entrée annoncée** : Enter long when VW-MACD crosses above its signal line with expanding volume bars. Short on bearish cross.
- **Sortie annoncée** : Exit on opposing VW-MACD/signal crossover. Take profit when VW histogram contracts 50% from peak.

## Données et formalisation

- **Indicateurs cités** : VW-MACD, VW Signal, VW Histogram, Volume
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
