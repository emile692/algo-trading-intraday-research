# POC Migration Trend

- **ID** : `poc-migration`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `auction-market`
- **Famille de recherche (provisoire)** : `volume_auction`
- **Rôle envisagé (provisoire)** : `alpha_or_feature`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/poc-migration) · [page MNQ](https://pinescriptforge.com/mnq/poc-migration/backtest)

## Description publiée

Trades in the direction of Point of Control migration across sessions for trend confirmation.

- **Entrée annoncée** : Enter long when today's developing POC is above yesterday's final POC AND price pulls back to today's POC. Opposite for short.
- **Sortie annoncée** : Exit when POC migration stalls (same level 2 consecutive sessions). Trail stop at prior session POC.

## Données et formalisation

- **Indicateurs cités** : POC, Prior POC, Migration Direction, Volume Profile
- **Horizons cités** : 30min, 1H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Calendrier de séance et horodatage New York à définir.
- **Donnée à vérifier** : Vérifier la nécessité de données tick, carnet ou ordres; pas de substitution OHLCV implicite.
- **Donnée à vérifier** : Distribution volume-prix nécessaire pour reproduire un vrai profil de marché.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
