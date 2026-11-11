# London-New York Overlap

- **ID** : `london-ny-overlap`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `time-based`
- **Famille de recherche (provisoire)** : `session_gap`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/london-ny-overlap) · [page MNQ](https://pinescriptforge.com/mnq/london-ny-overlap/backtest)

## Description publiée

Trades the high-liquidity London-NY session overlap for maximum volatility and volume.

- **Entrée annoncée** : Mark the high/low of 8:00–8:30 ET. Enter long on break above with volume surge. Short on break below. Require 2:1 range vs. pre-overlap range.
- **Sortie annoncée** : Exit at 11:30 ET (pre-overlap end). Trail stop at 1× ATR. Take profit at 2× the overlap opening range.

## Données et formalisation

- **Indicateurs cités** : Session Time, Opening Range, Volume, ATR
- **Horizons cités** : 5min, 15min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Calendrier de séance et horodatage New York à définir.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
