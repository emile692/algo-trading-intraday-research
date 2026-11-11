# Kalman Filter Trend

- **ID** : `kalman-filter-trend`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `statistical`
- **Famille de recherche (provisoire)** : `statistical`
- **Rôle envisagé (provisoire)** : `alpha_or_feature`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/kalman-filter-trend) · [page MNQ](https://pinescriptforge.com/mnq/kalman-filter-trend/backtest)

## Description publiée

Adaptive Kalman filter for noise-reduced trend estimation and signal generation.

- **Entrée annoncée** : Enter long when price crosses above Kalman estimate and slope is positive. Short when price crosses below with negative slope.
- **Sortie annoncée** : Exit when price re-crosses the Kalman line or slope flattens to near zero. Trail stop at Kalman estimate ± 1 ATR.

## Données et formalisation

- **Indicateurs cités** : Kalman Filter, Kalman Slope, ATR
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
