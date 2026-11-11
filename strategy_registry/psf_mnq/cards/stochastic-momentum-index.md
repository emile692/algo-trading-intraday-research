# Stochastic Momentum Index

- **ID** : `stochastic-momentum-index`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `momentum`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/stochastic-momentum-index) · [page MNQ](https://pinescriptforge.com/mnq/stochastic-momentum-index/backtest)

## Description publiée

Enhanced stochastic measuring distance from the midpoint of the high/low range.

- **Entrée annoncée** : Enter long when SMI crosses above its signal line below -40. Enter short when SMI crosses below its signal line above +40. Divergences provide highest-probability signals.
- **Sortie annoncée** : Exit at opposite extreme (+40 for longs, -40 for shorts) or on signal line cross against position.

## Données et formalisation

- **Indicateurs cités** : SMI 13,25, SMI Signal 2, EMA 50
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
