# Inside Bar Breakout

- **ID** : `inside-bar-breakout`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `breakout`
- **Famille de recherche (provisoire)** : `breakout_expansion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/inside-bar-breakout) · [page MNQ](https://pinescriptforge.com/mnq/inside-bar-breakout/backtest)

## Description publiée

Trades breakouts from inside bar patterns signaling volatility compression.

- **Entrée annoncée** : Identify an inside bar (high lower than previous high, low higher than previous low). Enter long on break above inside bar high, short on break below low.
- **Sortie annoncée** : Target 2× the inside bar range. Stop at the opposite side of the inside bar. Trail at 1× range after target 1.

## Données et formalisation

- **Indicateurs cités** : Inside Bar Detection, ATR, Volume
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
