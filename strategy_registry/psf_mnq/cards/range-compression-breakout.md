# Range Compression Breakout

- **ID** : `range-compression-breakout`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `breakout`
- **Famille de recherche (provisoire)** : `breakout_expansion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/range-compression-breakout) · [page MNQ](https://pinescriptforge.com/mnq/range-compression-breakout/backtest)

## Description publiée

Identifies narrowing price ranges (NR4/NR7 patterns) that precede explosive breakout moves.

- **Entrée annoncée** : Identify an NR7 bar (narrowest range of last 7 bars) that is also an inside bar. Enter long on break above the NR7 high, short on break below. Add confirmation if Bollinger Band width is at a 20-period low.
- **Sortie annoncée** : Target 2× the NR7 bar range from the breakout level. Stop at the opposite side of the NR7 bar. Trail at 1× the NR7 range after target 1 hit.

## Données et formalisation

- **Indicateurs cités** : NR4/NR7 Detection, Inside Bar, Bollinger Band Width, ATR
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
