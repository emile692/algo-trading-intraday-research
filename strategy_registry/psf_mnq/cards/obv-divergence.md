# OBV Divergence

- **ID** : `obv-divergence`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `volume`
- **Famille de recherche (provisoire)** : `volume_auction`
- **Rôle envisagé (provisoire)** : `alpha_or_feature`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/obv-divergence) · [page MNQ](https://pinescriptforge.com/mnq/obv-divergence/backtest)

## Description publiée

On-Balance Volume divergences revealing hidden accumulation or distribution.

- **Entrée annoncée** : Enter long when price makes a lower low but OBV makes a higher low (bullish divergence). Short on bearish divergence (price higher high, OBV lower high).
- **Sortie annoncée** : Target the previous swing high/low. Stop below/above the divergence low/high. Trail at 2× ATR.

## Données et formalisation

- **Indicateurs cités** : OBV, OBV Divergence, ATR
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
