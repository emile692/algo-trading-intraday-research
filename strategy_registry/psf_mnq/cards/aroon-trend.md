# Aroon Trend System

- **ID** : `aroon-trend`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `trend`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/aroon-trend) · [page MNQ](https://pinescriptforge.com/mnq/aroon-trend/backtest)

## Description publiée

Uses Aroon Up/Down to identify emerging trends and trend strength.

- **Entrée annoncée** : Enter long when Aroon Up crosses above Aroon Down and Aroon Up > 70. Short when Aroon Down crosses above Aroon Up and Aroon Down > 70.
- **Sortie annoncée** : Exit when Aroon for your direction drops below 50 or when a crossover occurs against position. Stop at 2× ATR.

## Données et formalisation

- **Indicateurs cités** : Aroon Up, Aroon Down, ATR
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
