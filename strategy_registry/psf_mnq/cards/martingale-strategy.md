# Martingale

- **ID** : `martingale-strategy`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `momentum`
- **Famille de recherche (provisoire)** : `money_management`
- **Rôle envisagé (provisoire)** : `exclude_as_alpha`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/martingale-strategy) · [page MNQ](https://pinescriptforge.com/mnq/martingale-strategy/backtest)

## Description publiée

Doubles position size after losses to recover on the next winner.

- **Entrée annoncée** : Take initial position based on any signal (e.g., MA crossover). After a loss, double position size and enter on the next signal in the same direction. Max 3 doublings.
- **Sortie annoncée** : Exit each trade at 1× risk target. Reset to base position after a winner. Emergency stop: halt after 4 consecutive losses or 5% daily drawdown.

## Données et formalisation

- **Indicateurs cités** : Position Sizing, Win/Loss Tracker, Daily P&L
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
