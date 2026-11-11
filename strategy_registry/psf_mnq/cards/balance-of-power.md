# Balance of Power

- **ID** : `balance-of-power`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `volume`
- **Famille de recherche (provisoire)** : `volume_auction`
- **Rôle envisagé (provisoire)** : `alpha_or_feature`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/balance-of-power) · [page MNQ](https://pinescriptforge.com/mnq/balance-of-power/backtest)

## Description publiée

Measures buying vs selling pressure using the close position within the range.

- **Entrée annoncée** : Enter long when smoothed BOP crosses above zero from deeply negative. Enter short when BOP crosses below zero from deeply positive. Divergence entries for highest probability.
- **Sortie annoncée** : Exit when BOP crosses zero against position or diverges from price. Trail at 20 EMA.

## Données et formalisation

- **Indicateurs cités** : Balance of Power, BOP SMA 14, EMA 20
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
