# VWAP Anchored

- **ID** : `vwap-anchored`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `volume`
- **Famille de recherche (provisoire)** : `volume_auction`
- **Rôle envisagé (provisoire)** : `alpha_or_feature`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/vwap-anchored) · [page MNQ](https://pinescriptforge.com/mnq/vwap-anchored/backtest)

## Description publiée

Anchors VWAP to significant events (earnings, gaps, highs/lows) for dynamic S/R.

- **Entrée annoncée** : Enter long on bounce from anchored VWAP at swing lows. Short on rejection from anchored VWAP at swing highs.
- **Sortie annoncée** : Target the next anchored VWAP level. Stop 1× ATR beyond the anchored VWAP.

## Données et formalisation

- **Indicateurs cités** : Anchored VWAP, Volume, Swing Points
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
