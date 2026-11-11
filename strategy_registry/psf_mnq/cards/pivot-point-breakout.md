# Pivot Point Breakout

- **ID** : `pivot-point-breakout`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `breakout`
- **Famille de recherche (provisoire)** : `breakout_expansion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/pivot-point-breakout) · [page MNQ](https://pinescriptforge.com/mnq/pivot-point-breakout/backtest)

## Description publiée

Trades breakouts and rejections at classic floor trader pivot levels.

- **Entrée annoncée** : Enter long on break above R1 with volume surge. Enter short on break below S1. Mean reversion: buy at S2 rejection, sell at R2 rejection.
- **Sortie annoncée** : Breakout targets: next pivot level (R1→R2, S1→S2). Rejection targets: PP (central pivot). Stop beyond the pivot level.

## Données et formalisation

- **Indicateurs cités** : Pivot Points, R1/R2/R3, S1/S2/S3, Volume
- **Horizons cités** : 5min, 15min, 1H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).
- **Nom partagé avec** : `pivot-breakout`. Dédupliquer la logique avant expérimentation.

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
