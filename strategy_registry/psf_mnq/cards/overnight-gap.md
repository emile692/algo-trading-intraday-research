# Overnight Gap Strategy

- **ID** : `overnight-gap`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `time-based`
- **Famille de recherche (provisoire)** : `session_gap`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/overnight-gap) · [page MNQ](https://pinescriptforge.com/mnq/overnight-gap/backtest)

## Description publiée

Trades the gap between previous close and current open for gap-fill or gap-and-go setups.

- **Entrée annoncée** : Gap fill: fade gaps < 0.5% with low volume (enter against gap direction). Gap & go: trade with gaps > 1% that have strong volume at open.
- **Sortie annoncée** : Fill trades: target prior close. Go trades: trail at VWAP. Both: stop at 50% of gap distance against position.

## Données et formalisation

- **Indicateurs cités** : Gap Size, Gap Direction, Volume at Open, VWAP
- **Horizons cités** : 1min, 5min, 15min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Calendrier de séance et horodatage New York à définir.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
