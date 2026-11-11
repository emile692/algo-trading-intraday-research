# Momentum Oscillator

- **ID** : `momentum-oscillator`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `momentum`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/momentum-oscillator) · [page MNQ](https://pinescriptforge.com/mnq/momentum-oscillator/backtest)

## Description publiée

Raw price momentum measurement for detecting acceleration and deceleration.

- **Entrée annoncée** : Enter long when Momentum crosses above zero with rising slope and price above 50 EMA. Short on cross below zero with declining slope.
- **Sortie annoncée** : Exit when Momentum slope flattens or reverses. Stop at 2× ATR from entry.

## Données et formalisation

- **Indicateurs cités** : Momentum 14, EMA 50, ATR
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
