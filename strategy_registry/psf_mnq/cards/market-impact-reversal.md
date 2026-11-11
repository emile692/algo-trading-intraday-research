# Market Impact Reversal

- **ID** : `market-impact-reversal`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `microstructure`
- **Famille de recherche (provisoire)** : `order_flow`
- **Rôle envisagé (provisoire)** : `data_gated_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/market-impact-reversal) · [page MNQ](https://pinescriptforge.com/mnq/market-impact-reversal/backtest)

## Description publiée

Fades large aggressive orders that temporarily push price beyond fair value.

- **Entrée annoncée** : Enter fade when volume spike > 3× average occurs AND price immediately rejects (wick > 50% of bar). Fade the spike direction.
- **Sortie annoncée** : Target pre-impact price level. Stop at impact extreme + 2 ticks. Time-based exit after 5 minutes if target not reached.

## Données et formalisation

- **Indicateurs cités** : Volume Spike, Price Rejection, VWAP, Book Depth
- **Horizons cités** : 1min, 5min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Vérifier la nécessité de données tick, carnet ou ordres; pas de substitution OHLCV implicite.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
