# Rising / Falling Wedge

- **ID** : `wedge-pattern`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `pattern-recognition`
- **Famille de recherche (provisoire)** : `chart_pattern`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/wedge-pattern) · [page MNQ](https://pinescriptforge.com/mnq/wedge-pattern/backtest)

## Description publiée

Trades rising wedge reversals and falling wedge breakouts for high-probability entries.

- **Entrée annoncée** : Enter short on rising wedge breakdown with volume expansion. Long on falling wedge breakout. Volume should diverge (decline) during wedge formation.
- **Sortie annoncée** : Target: wedge starting point (full reversal). Stop above/below wedge extreme. Partial at 50% of wedge height.

## Données et formalisation

- **Indicateurs cités** : Wedge Pattern, Volume Divergence, Trendlines, ATR
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
