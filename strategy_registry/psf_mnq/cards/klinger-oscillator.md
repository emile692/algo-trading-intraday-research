# Klinger Oscillator

- **ID** : `klinger-oscillator`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `volume`
- **Famille de recherche (provisoire)** : `volume_auction`
- **Rôle envisagé (provisoire)** : `alpha_or_feature`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/klinger-oscillator) · [page MNQ](https://pinescriptforge.com/mnq/klinger-oscillator/backtest)

## Description publiée

Compares volume flowing in and out of a security to predict reversals.

- **Entrée annoncée** : Enter long when Klinger crosses above its signal line below zero. Enter short when it crosses below signal above zero.
- **Sortie annoncée** : Exit on opposing signal crossover. Secondary exit when Klinger reaches extreme levels (95th percentile).

## Données et formalisation

- **Indicateurs cités** : Klinger Oscillator, Klinger Signal, Volume
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
