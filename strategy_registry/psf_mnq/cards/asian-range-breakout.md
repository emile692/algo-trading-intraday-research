# Asian Range Breakout

- **ID** : `asian-range-breakout`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `breakout`
- **Famille de recherche (provisoire)** : `breakout_expansion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/asian-range-breakout) · [page MNQ](https://pinescriptforge.com/mnq/asian-range-breakout/backtest)

## Description publiée

Measures the Asian session consolidation range and trades the breakout in London/NY.

- **Entrée annoncée** : Calculate the Asian session range (18:00-02:00 ET). Enter long on break above + ATR filter. Short on break below. Must trigger during London or NY session.
- **Sortie annoncée** : Target 2× Asian range. Stop at 0.5× Asian range inside the channel. Trail after first target.

## Données et formalisation

- **Indicateurs cités** : Asian Range, ATR Filter, Session Timer
- **Horizons cités** : 5min, 15min, 1H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Calendrier de séance et horodatage New York à définir.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
