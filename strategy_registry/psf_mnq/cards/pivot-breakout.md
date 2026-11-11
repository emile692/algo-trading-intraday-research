# Pivot Point Breakout

- **ID** : `pivot-breakout`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `breakout`
- **Famille de recherche (provisoire)** : `breakout_expansion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/pivot-breakout) · [page MNQ](https://pinescriptforge.com/mnq/pivot-breakout/backtest)

## Description publiée

Trades breakouts through daily pivot points for intraday directional trades.

- **Entrée annoncée** : Enter long on break above R1 with above-average volume. Short on break below S1. Close must be beyond the level, not just a wick.
- **Sortie annoncée** : Target R2 (long) or S2 (short). Stop at the pivot (P). Partial exit at 50% of the R1-R2 distance.

## Données et formalisation

- **Indicateurs cités** : Pivot Points, R1/R2/S1/S2, Volume, ATR
- **Horizons cités** : 15min, 1H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).
- **Nom partagé avec** : `pivot-point-breakout`. Dédupliquer la logique avant expérimentation.

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
