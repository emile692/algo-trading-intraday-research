# Envelope Channel Reversion

- **ID** : `envelope-channel-reversion`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `mean-reversion`
- **Famille de recherche (provisoire)** : `mean_reversion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/envelope-channel-reversion) · [page MNQ](https://pinescriptforge.com/mnq/envelope-channel-reversion/backtest)

## Description publiée

Trades reversion from percentage-based envelope channels around a moving average.

- **Entrée annoncée** : Long at the lower envelope (-2% from 20 SMA). Short at the upper envelope (+2% from 20 SMA).
- **Sortie annoncée** : Target the 20 SMA center line. Stop at 1.5× the envelope width beyond entry.

## Données et formalisation

- **Indicateurs cités** : Price Envelopes, SMA 20, RSI
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
