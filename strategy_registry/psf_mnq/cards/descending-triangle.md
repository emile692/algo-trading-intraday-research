# Descending Triangle Breakdown

- **ID** : `descending-triangle`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `pattern-recognition`
- **Famille de recherche (provisoire)** : `chart_pattern`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/descending-triangle) · [page MNQ](https://pinescriptforge.com/mnq/descending-triangle/backtest)

## Description publiée

Trades bearish descending triangle breakdowns with flat support and falling resistance.

- **Entrée annoncée** : Enter short on close below flat support with volume expansion. Require at least 3 touches on each line.
- **Sortie annoncée** : Target: triangle height subtracted from breakdown level. Stop above last descending resistance touch.

## Données et formalisation

- **Indicateurs cités** : Triangle Pattern, Support Level, Descending Resistance, Volume
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
