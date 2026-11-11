# Initial Balance Breakout

- **ID** : `initial-balance-breakout`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `auction-market`
- **Famille de recherche (provisoire)** : `volume_auction`
- **Rôle envisagé (provisoire)** : `alpha_or_feature`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/initial-balance-breakout) · [page MNQ](https://pinescriptforge.com/mnq/initial-balance-breakout/backtest)

## Description publiée

Trades breakouts from the first hour's range using auction market theory principles.

- **Entrée annoncée** : Enter long on close above IB high with volume confirmation. Short on close below IB low. Require bar to close outside IB, not just wick.
- **Sortie annoncée** : First target at 1.5× IB range extension. Second target at 2× extension. Stop at IB midpoint (50% of range).

## Données et formalisation

- **Indicateurs cités** : Initial Balance High/Low, IB Range, Volume, Market Profile
- **Horizons cités** : 5min, 15min, 30min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Distribution volume-prix nécessaire pour reproduire un vrai profil de marché.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
