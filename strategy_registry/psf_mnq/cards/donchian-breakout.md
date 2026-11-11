# Donchian Channel Breakout

- **ID** : `donchian-breakout`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `breakout`
- **Famille de recherche (provisoire)** : `breakout_expansion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/donchian-breakout) · [page MNQ](https://pinescriptforge.com/mnq/donchian-breakout/backtest)

## Description publiée

Turtle Trading-inspired breakout system using Donchian channels.

- **Entrée annoncée** : Enter long when price breaks above the 20-period Donchian high. Enter short on break below 20-period low.
- **Sortie annoncée** : Exit long on break below the 10-period Donchian low. Exit short on break above 10-period high. ATR-based position sizing.

## Données et formalisation

- **Indicateurs cités** : Donchian Channel 20, Donchian Channel 10, ATR
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
