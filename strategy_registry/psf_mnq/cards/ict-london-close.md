# ICT London Close

- **ID** : `ict-london-close`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `smart-money`
- **Famille de recherche (provisoire)** : `price_action_reversal`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/ict-london-close) · [page MNQ](https://pinescriptforge.com/mnq/ict-london-close/backtest)

## Description publiée

Fades the London session move during the London close window (10-12 EST).

- **Entrée annoncée** : Fade the London session direction during 10-12 EST. Short if London rallied, long if it sold off.
- **Sortie annoncée** : Target 50% retracement of the London move. Stop beyond the London session extreme.

## Données et formalisation

- **Indicateurs cités** : London Session H/L, Session VWAP, Time Filter
- **Horizons cités** : 5min, 15min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Calendrier de séance et horodatage New York à définir.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
