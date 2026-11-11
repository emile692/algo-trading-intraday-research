# Ascending Triangle Breakout

- **ID** : `ascending-triangle`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `pattern-recognition`
- **Famille de recherche (provisoire)** : `chart_pattern`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/ascending-triangle) · [page MNQ](https://pinescriptforge.com/mnq/ascending-triangle/backtest)

## Description publiée

Trades bullish ascending triangle breakouts with flat resistance and rising support.

- **Entrée annoncée** : Enter long on close above flat resistance with volume > 2× average. Confirm at least 3 touches on both lines.
- **Sortie annoncée** : Target: triangle height added to breakout level. Stop below the last rising support touch. Trail at ascending support line.

## Données et formalisation

- **Indicateurs cités** : Triangle Pattern, Resistance Level, Rising Support, Volume
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
