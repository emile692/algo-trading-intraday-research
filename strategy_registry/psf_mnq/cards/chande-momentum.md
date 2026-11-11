# Chande Momentum Oscillator

- **ID** : `chande-momentum`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `momentum`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/chande-momentum) · [page MNQ](https://pinescriptforge.com/mnq/chande-momentum/backtest)

## Description publiée

Tushar Chande's oscillator measuring both up and down momentum for balanced signals.

- **Entrée annoncée** : Enter long when CMO crosses above +50. Enter short when CMO crosses below -50. Use 200 SMA as trend filter.
- **Sortie annoncée** : Exit long when CMO drops below +50. Exit short when CMO rises above -50. Stop at recent swing.

## Données et formalisation

- **Indicateurs cités** : CMO, SMA 200
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
