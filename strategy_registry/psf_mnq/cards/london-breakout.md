# London Breakout

- **ID** : `london-breakout`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `breakout`
- **Famille de recherche (provisoire)** : `breakout_expansion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/london-breakout) · [page MNQ](https://pinescriptforge.com/mnq/london-breakout/backtest)

## Description publiée

Trades the breakout from the pre-London Asian session range during the London open.

- **Entrée annoncée** : Mark the high and low from 00:00-08:00 GMT (Asian range). Enter long on break above Asian high after 08:00 GMT. Short on break below Asian low.
- **Sortie annoncée** : Target 1.5× the Asian range. Stop at the opposite side of the Asian range. Close by 16:00 GMT.

## Données et formalisation

- **Indicateurs cités** : Asian Range, Session Timer, ATR
- **Horizons cités** : 5min, 15min, 1H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Calendrier de séance et horodatage New York à définir.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
