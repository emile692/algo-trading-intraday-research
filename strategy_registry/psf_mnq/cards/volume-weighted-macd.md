# Volume Weighted MACD

- **ID** : `volume-weighted-macd`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `volume`
- **Famille de recherche (provisoire)** : `volume_auction`
- **Rôle envisagé (provisoire)** : `alpha_or_feature`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/volume-weighted-macd) · [page MNQ](https://pinescriptforge.com/mnq/volume-weighted-macd/backtest)

## Description publiée

MACD calculated on volume-weighted price for superior signal quality.

- **Entrée annoncée** : Enter long when VW-MACD crosses above its signal line with increasing volume. Short on bearish crossover with volume confirmation.
- **Sortie annoncée** : Exit on opposing crossover. Additional exit when volume drops below 20-period average (conviction fading).

## Données et formalisation

- **Indicateurs cités** : VW-MACD, VW-MACD Signal, Volume Average
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
