# Detrended Price Oscillator

- **ID** : `dpo-cycle`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `momentum`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/dpo-cycle) · [page MNQ](https://pinescriptforge.com/mnq/dpo-cycle/backtest)

## Description publiée

Removes trend to reveal underlying price cycles for timing entries.

- **Entrée annoncée** : Enter long when DPO crosses above zero from a trough. Enter short when DPO crosses below zero from a peak.
- **Sortie annoncée** : Exit when DPO reaches the opposite extreme or crosses zero against position. Stop at recent swing.

## Données et formalisation

- **Indicateurs cités** : DPO, SMA 20, Cycle Length
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
