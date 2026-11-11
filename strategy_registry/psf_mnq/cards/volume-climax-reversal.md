# Volume Climax Reversal

- **ID** : `volume-climax-reversal`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `volume`
- **Famille de recherche (provisoire)** : `volume_auction`
- **Rôle envisagé (provisoire)** : `alpha_or_feature`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/volume-climax-reversal) · [page MNQ](https://pinescriptforge.com/mnq/volume-climax-reversal/backtest)

## Description publiée

Trades reversals after volume climax events (3σ+ volume spikes).

- **Entrée annoncée** : Fade the direction of a volume climax bar (>3σ volume). Wait for next-bar confirmation.
- **Sortie annoncée** : Target 50% retracement of the climax bar range. Stop beyond the climax bar extreme.

## Données et formalisation

- **Indicateurs cités** : Volume, Volume StdDev, Climax Detection
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
