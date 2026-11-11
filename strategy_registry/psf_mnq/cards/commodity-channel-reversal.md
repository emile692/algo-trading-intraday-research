# Commodity Channel Reversal

- **ID** : `commodity-channel-reversal`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `mean-reversion`
- **Famille de recherche (provisoire)** : `mean_reversion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/commodity-channel-reversal) · [page MNQ](https://pinescriptforge.com/mnq/commodity-channel-reversal/backtest)

## Description publiée

CCI extreme reversal strategy entering on pullback from +200/-200 levels.

- **Entrée annoncée** : Enter long when CCI drops below -200 then crosses back above -100 (exhaustion confirmed). Opposite for short.
- **Sortie annoncée** : Target CCI returning to zero. Stop if CCI makes a new extreme beyond the entry extreme.

## Données et formalisation

- **Indicateurs cités** : CCI 20, SMA 50, ATR
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
