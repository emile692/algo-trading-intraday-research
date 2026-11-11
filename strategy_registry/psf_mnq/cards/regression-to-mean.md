# Regression to Mean

- **ID** : `regression-to-mean`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `mean-reversion`
- **Famille de recherche (provisoire)** : `mean_reversion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/regression-to-mean) · [page MNQ](https://pinescriptforge.com/mnq/regression-to-mean/backtest)

## Description publiée

Linear regression-based mean reversion when price deviates beyond normal ranges.

- **Entrée annoncée** : Enter long when price is below the regression line by 2σ with regression slope positive. Short when 2σ above with negative slope.
- **Sortie annoncée** : Target the regression line. Stop at 3σ deviation. Partial at 1σ.

## Données et formalisation

- **Indicateurs cités** : Linear Regression, Standard Deviation Bands, R-Squared
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
