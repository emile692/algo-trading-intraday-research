# TEMA Crossover

- **ID** : `tema-crossover`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `trend`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/tema-crossover) · [page MNQ](https://pinescriptforge.com/mnq/tema-crossover/backtest)

## Description publiée

Triple Exponential MA crossover for ultra-responsive trend signals.

- **Entrée annoncée** : Enter long when fast TEMA (8) crosses above slow TEMA (21). Short on bearish crossover. ADX > 20 as trend filter.
- **Sortie annoncée** : Exit on opposing TEMA crossover. Trail at slow TEMA. Stop at 1.5× ATR.

## Données et formalisation

- **Indicateurs cités** : TEMA 8, TEMA 21, ADX, ATR
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
