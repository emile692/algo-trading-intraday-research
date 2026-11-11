# Parabolic SAR Trend

- **ID** : `parabolic-sar-trend`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `trend`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/parabolic-sar-trend) · [page MNQ](https://pinescriptforge.com/mnq/parabolic-sar-trend/backtest)

## Description publiée

Uses Parabolic SAR dots for trend direction and trailing stops.

- **Entrée annoncée** : Enter long when SAR dots flip below price. Enter short when dots flip above price. Confirm with 50 EMA direction.
- **Sortie annoncée** : Exit when SAR dots flip against position. The SAR itself serves as the trailing stop — no manual adjustment needed.

## Données et formalisation

- **Indicateurs cités** : Parabolic SAR, EMA 50
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
