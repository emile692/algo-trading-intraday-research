# Anti-Martingale

- **ID** : `anti-martingale`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `momentum`
- **Famille de recherche (provisoire)** : `money_management`
- **Rôle envisagé (provisoire)** : `exclude_as_alpha`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/anti-martingale) · [page MNQ](https://pinescriptforge.com/mnq/anti-martingale/backtest)

## Description publiée

Increases position size after wins to maximize winning streaks.

- **Entrée annoncée** : Start with base position. After each winner, increase size by 50% (up to 3× base). After each loser, reset to base size. Use MA crossover for direction.
- **Sortie annoncée** : Fixed target at 1.5× risk per trade. Stop at 1× risk. Reset position size to base after any loss.

## Données et formalisation

- **Indicateurs cités** : Position Sizing, Streak Tracker, EMA 20/50
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
