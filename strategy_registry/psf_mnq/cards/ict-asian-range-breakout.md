# ICT Asian Range Breakout

- **ID** : `ict-asian-range-breakout`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `smart-money`
- **Famille de recherche (provisoire)** : `price_action_reversal`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/ict-asian-range-breakout) · [page MNQ](https://pinescriptforge.com/mnq/ict-asian-range-breakout/backtest)

## Description publiée

Trades the breakout from the Asian session range during London or NY killzones.

- **Entrée annoncée** : Mark the Asian session high and low (8 PM–12 AM ET). During London or NY killzone, enter long on sustained break above Asian high or short on break below Asian low, confirmed by displacement candle.
- **Sortie annoncée** : Target 1.5× the Asian range projected from the breakout. Stop inside the Asian range at the midpoint. Time exit at end of NY session.

## Données et formalisation

- **Indicateurs cités** : Asian Range High/Low, Killzone Timer, Displacement Detection
- **Horizons cités** : 5min, 15min, 1H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Calendrier de séance et horodatage New York à définir.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
