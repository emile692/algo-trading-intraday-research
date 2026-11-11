# TTM Squeeze Momentum

- **ID** : `squeeze-momentum`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `volatility`
- **Famille de recherche (provisoire)** : `volatility_regime`
- **Rôle envisagé (provisoire)** : `feature_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/squeeze-momentum) · [page MNQ](https://pinescriptforge.com/mnq/squeeze-momentum/backtest)

## Description publiée

John Carter's Squeeze indicator combining Bollinger Bands and Keltner Channels.

- **Entrée annoncée** : Wait for squeeze (BBs inside KCs). Enter long when squeeze fires and momentum histogram is positive and rising. Enter short on negative histogram.
- **Sortie annoncée** : Exit when momentum histogram starts declining (momentum fading). Trail at the Keltner Channel midline.

## Données et formalisation

- **Indicateurs cités** : Bollinger Bands, Keltner Channels, Momentum Histogram
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
