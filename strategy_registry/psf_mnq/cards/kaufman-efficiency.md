# Kaufman Efficiency

- **ID** : `kaufman-efficiency`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `trend`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/kaufman-efficiency) · [page MNQ](https://pinescriptforge.com/mnq/kaufman-efficiency/backtest)

## Description publiée

Adaptive moving average that speeds up in trends and slows in chop.

- **Entrée annoncée** : Enter long when KAMA turns up and price > KAMA. Short when KAMA turns down. ER > 0.6 = strong signal.
- **Sortie annoncée** : Exit when KAMA flattens (slope < threshold) or price crosses KAMA. Stop at 1.5× ATR.

## Données et formalisation

- **Indicateurs cités** : KAMA, Efficiency Ratio, ATR
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
