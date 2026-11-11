# Wyckoff Spring

- **ID** : `wyckoff-spring`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `wyckoff`
- **Famille de recherche (provisoire)** : `price_action_reversal`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/wyckoff-spring) · [page MNQ](https://pinescriptforge.com/mnq/wyckoff-spring/backtest)

## Description publiée

Identifies spring setups where price briefly breaks below support to trap sellers.

- **Entrée annoncée** : Enter long when price breaks below the trading range support, then quickly reverses back above it within 1-3 bars. Volume should spike on the spring and increase on the recovery.
- **Sortie annoncée** : Target the top of the trading range (Creek/resistance). Stop below the spring low. Scale out at Sign of Strength (SOS) bars.

## Données et formalisation

- **Indicateurs cités** : Volume, Support/Resistance, Price Spread, Effort vs Result
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
