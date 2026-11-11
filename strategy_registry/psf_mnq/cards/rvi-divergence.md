# Relative Vigor Index

- **ID** : `rvi-divergence`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `momentum`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/rvi-divergence) · [page MNQ](https://pinescriptforge.com/mnq/rvi-divergence/backtest)

## Description publiée

Measures conviction of price moves by comparing close vs open relative to range.

- **Entrée annoncée** : Enter long when RVI crosses above its signal line in oversold territory. Short when RVI crosses below signal in overbought territory.
- **Sortie annoncée** : Exit on opposing RVI/signal crossover. Take profit at 2× risk or when RVI reaches extreme.

## Données et formalisation

- **Indicateurs cités** : RVI, RVI Signal Line
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).
- **Nom partagé avec** : `relative-vigor-index`. Dédupliquer la logique avant expérimentation.

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
