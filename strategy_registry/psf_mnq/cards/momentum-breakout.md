# Momentum Breakout

- **ID** : `momentum-breakout`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `breakout`
- **Famille de recherche (provisoire)** : `breakout_expansion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/momentum-breakout) · [page MNQ](https://pinescriptforge.com/mnq/momentum-breakout/backtest)

## Description publiée

Combines price breakouts with momentum confirmation for filtered entries.

- **Entrée annoncée** : Enter long on 20-bar high breakout with RSI > 60 and ADX > 25. Enter short on 20-bar low breakdown with RSI < 40 and ADX > 25.
- **Sortie annoncée** : Trail at 2× ATR from entry. Time stop at 20 bars if target not reached. Exit on RSI divergence.

## Données et formalisation

- **Indicateurs cités** : 20-bar High/Low, RSI 14, ADX 14
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
