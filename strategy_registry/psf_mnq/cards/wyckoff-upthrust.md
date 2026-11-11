# Wyckoff Upthrust

- **ID** : `wyckoff-upthrust`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `wyckoff`
- **Famille de recherche (provisoire)** : `price_action_reversal`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/wyckoff-upthrust) · [page MNQ](https://pinescriptforge.com/mnq/wyckoff-upthrust/backtest)

## Description publiée

Detects upthrust moves above resistance to trap buyers before distribution.

- **Entrée annoncée** : Enter short when price breaks above the trading range resistance, then quickly falls back below within 1-3 bars. Volume should spike and price spread should narrow.
- **Sortie annoncée** : Target the bottom of the trading range (support). Stop above the upthrust high. Trail at Signs of Weakness (SOW).

## Données et formalisation

- **Indicateurs cités** : Volume, Support/Resistance, Price Spread, Effort vs Result
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
