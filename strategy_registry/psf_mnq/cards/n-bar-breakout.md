# N-Bar Breakout

- **ID** : `n-bar-breakout`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `breakout`
- **Famille de recherche (provisoire)** : `breakout_expansion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/n-bar-breakout) · [page MNQ](https://pinescriptforge.com/mnq/n-bar-breakout/backtest)

## Description publiée

Breaks above the highest high or below the lowest low of the last N bars.

- **Entrée annoncée** : Enter long when price breaks above the highest high of the last N bars (default N=20). Enter short on break below lowest low.
- **Sortie annoncée** : Exit on break of the opposite N/2 bar extreme. Trail stop at the N/2 bar high/low.

## Données et formalisation

- **Indicateurs cités** : N-Bar High, N-Bar Low, ATR
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
