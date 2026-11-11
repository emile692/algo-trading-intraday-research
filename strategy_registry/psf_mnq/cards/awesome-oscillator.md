# Awesome Oscillator

- **ID** : `awesome-oscillator`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `momentum`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/awesome-oscillator) · [page MNQ](https://pinescriptforge.com/mnq/awesome-oscillator/backtest)

## Description publiée

Bill Williams' momentum oscillator using midpoint price difference for trend momentum.

- **Entrée annoncée** : Saucer buy: AO above zero, red bar followed by green bar (momentum resumption). Zero cross: AO crosses above zero from below.
- **Sortie annoncée** : Exit on opposing signal or when AO histogram shows 3 consecutive contracting bars. Stop at swing low/high.

## Données et formalisation

- **Indicateurs cités** : Awesome Oscillator, AO Histogram
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
