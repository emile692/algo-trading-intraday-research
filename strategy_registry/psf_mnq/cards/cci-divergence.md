# CCI Divergence

- **ID** : `cci-divergence`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `mean-reversion`
- **Famille de recherche (provisoire)** : `mean_reversion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/cci-divergence) · [page MNQ](https://pinescriptforge.com/mnq/cci-divergence/backtest)

## Description publiée

Trades commodity channel index divergences for high-probability reversals.

- **Entrée annoncée** : Enter long on bullish CCI divergence below -200 (price lower low, CCI higher low). Enter short on bearish divergence above +200. Dual-period confirmation for highest conviction.
- **Sortie annoncée** : Exit when CCI returns to the ±100 zone. Partial exit at zero line. Stop beyond the divergence extreme.

## Données et formalisation

- **Indicateurs cités** : CCI 14, CCI 50, Price Swings
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
