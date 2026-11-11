# Hull Moving Average

- **ID** : `hull-moving-average`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `trend`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/hull-moving-average) · [page MNQ](https://pinescriptforge.com/mnq/hull-moving-average/backtest)

## Description publiée

Ultra-responsive moving average that reduces lag while maintaining smoothness.

- **Entrée annoncée** : Enter long when HMA changes from declining to rising (color change). Enter short when HMA starts declining. Confirm with price above/below HMA.
- **Sortie annoncée** : Exit on HMA direction change. Optional: use a faster HMA for exit and slower HMA for entry.

## Données et formalisation

- **Indicateurs cités** : Hull MA, HMA Direction
- **Horizons cités** : 5min, 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
