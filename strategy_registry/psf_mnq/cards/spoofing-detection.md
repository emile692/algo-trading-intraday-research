# Spoofing Detection Fade

- **ID** : `spoofing-detection`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `microstructure`
- **Famille de recherche (provisoire)** : `order_flow`
- **Rôle envisagé (provisoire)** : `data_gated_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/spoofing-detection) · [page MNQ](https://pinescriptforge.com/mnq/spoofing-detection/backtest)

## Description publiée

Identifies and fades spoofed orders that create false directional signals.

- **Entrée annoncée** : Detect spoof (large order placed then canceled within seconds). Fade the spoof direction — if spoof was on bid, go short. If on ask, go long.
- **Sortie annoncée** : Quick 3-5 tick target. Stop at 5 ticks. Exit immediately if actual volume confirms the spoofed direction.

## Données et formalisation

- **Indicateurs cités** : Order Flow, Cancel Rate, Volume Delta, Book Pressure
- **Horizons cités** : 1min, 5min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Vérifier la nécessité de données tick, carnet ou ordres; pas de substitution OHLCV implicite.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
