# Order Flow Delta Divergence

- **ID** : `order-flow-delta-divergence`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `volume`
- **Famille de recherche (provisoire)** : `volume_auction`
- **Rôle envisagé (provisoire)** : `alpha_or_feature`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/order-flow-delta-divergence) · [page MNQ](https://pinescriptforge.com/mnq/order-flow-delta-divergence/backtest)

## Description publiée

Detects divergence between price and buying/selling pressure delta for early reversals.

- **Entrée annoncée** : Enter short when price makes a higher high but cumulative delta makes a lower high (bearish delta divergence) with volume above average. Enter long on bullish delta divergence (price lower low, delta higher low).
- **Sortie annoncée** : Target the prior delta equilibrium zone (where delta crossed zero). Stop above the divergence high/low. Close position if delta re-confirms the prior trend direction.

## Données et formalisation

- **Indicateurs cités** : Cumulative Delta, Volume Delta, Divergence Detection, Volume MA
- **Horizons cités** : 5min, 15min, 1H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
