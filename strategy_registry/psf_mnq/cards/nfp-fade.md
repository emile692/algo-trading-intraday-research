# NFP Fade

- **ID** : `nfp-fade`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `seasonal`
- **Famille de recherche (provisoire)** : `calendar`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/nfp-fade) · [page MNQ](https://pinescriptforge.com/mnq/nfp-fade/backtest)

## Description publiée

Fades the initial reaction to Non-Farm Payrolls for a reversal trade.

- **Entrée annoncée** : Wait 15 minutes after NFP release. Fade the direction of the initial spike with a limit order at the 50% retracement of the spike.
- **Sortie annoncée** : Target the pre-release price level. Stop at 127% extension of the initial spike. Hold up to 2 hours post-release.

## Données et formalisation

- **Indicateurs cités** : Economic Calendar, NFP, 15min Chart, ATR
- **Horizons cités** : 5min, 15min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Calendrier de séance et horodatage New York à définir.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
