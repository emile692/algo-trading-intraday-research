# Ease of Movement

- **ID** : `ease-of-movement`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `volume`
- **Famille de recherche (provisoire)** : `volume_auction`
- **Rôle envisagé (provisoire)** : `alpha_or_feature`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/ease-of-movement) · [page MNQ](https://pinescriptforge.com/mnq/ease-of-movement/backtest)

## Description publiée

Measures how easily price moves relative to volume for trend confirmation.

- **Entrée annoncée** : Enter long when smoothed EMV crosses above zero and is rising. Enter short when EMV crosses below zero and is falling.
- **Sortie annoncée** : Exit when EMV crosses zero against position or when EMV diverges from price direction. Trail at 2× ATR.

## Données et formalisation

- **Indicateurs cités** : Ease of Movement, EMV SMA 14, Volume
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
