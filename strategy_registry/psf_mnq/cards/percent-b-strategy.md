# Percent B (%B) Strategy

- **ID** : `percent-b-strategy`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `volatility`
- **Famille de recherche (provisoire)** : `volatility_regime`
- **Rôle envisagé (provisoire)** : `feature_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/percent-b-strategy) · [page MNQ](https://pinescriptforge.com/mnq/percent-b-strategy/backtest)

## Description publiée

Quantifies price position within Bollinger Bands for precise entry timing.

- **Entrée annoncée** : Enter long when %B drops below 0.0 (below lower band) and turns upward. Enter short when %B exceeds 1.0 and turns downward. Breakout mode: enter when %B sustains above 0.8.
- **Sortie annoncée** : Mean reversion: target %B = 0.5 (middle band). Breakout: trail while %B stays above 0.5. Stop at %B beyond entry extreme.

## Données et formalisation

- **Indicateurs cités** : %B, Bollinger Bands, SMA 20, ATR
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
