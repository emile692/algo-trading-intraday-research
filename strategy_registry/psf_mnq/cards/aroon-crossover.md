# Aroon Crossover

- **ID** : `aroon-crossover`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `trend`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/aroon-crossover) · [page MNQ](https://pinescriptforge.com/mnq/aroon-crossover/backtest)

## Description publiée

Detects emerging trends using the time since the highest high and lowest low.

- **Entrée annoncée** : Enter long when Aroon Up crosses above 70 and Aroon Down drops below 30. Short on opposite configuration.
- **Sortie annoncée** : Exit when Aroon lines converge (both near 50) — trend is exhausted. Stop at 2× ATR.

## Données et formalisation

- **Indicateurs cités** : Aroon Up, Aroon Down, ATR
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
