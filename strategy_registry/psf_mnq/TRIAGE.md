# Triage initial du registre MNQ

Le [triage complet](triage.csv) couvre les 185 fiches. Les regroupements sont des **clusters de mécanismes provisoires**, pas une affirmation que deux règles sont identiques. Le classement du site et ses métriques ne déterminent aucune priorité.

## Univers décidé ensemble

Le fichier [collaborative_universe.csv](collaborative_universe.csv) contient actuellement **134 stratégies réparties dans 13 grandes familles**. Les règles retenues sont :

- stratégies compatibles avec les seules données OHLCV disponibles;
- aucun filtre sur l'horizon, y compris 1 h, 4 h et daily;
- exclusion des 20 fiches explicitement ICT;
- exclusion de `Market Structure Shift`, `Liquidity Sweep Reversal`, `Supply & Demand Zones` et des trois stratégies Wyckoff voisines;
- exclusion de `Grid Trading`, `Martingale` et `Anti-Martingale`, qui sont des méthodes de gestion du risque plutôt que des sources d'alpha.

Les fiches exclues restent dans le registre historique des 185 pour conserver la provenance du tri.

Le tri suivant par fonction dans l'architecture de recherche est documenté
dans [ARCHITECTURE_ROLES.md](ARCHITECTURE_ROLES.md) et détaillé dans
[architecture_roles.csv](architecture_roles.csv). Il conserve les 134
stratégies et leur attribue le statut `draft_to_review`.

## Données réellement disponibles

- MNQ continu 1 minute : 2 401 697 barres horodatées New York, du 5 mai 2019 au 20 mars 2026 à 09:29; OHLCV, `instrument_id`, aucun doublon d'horodatage observé.
- Les minutes sans volume peuvent manquer; les données sont des barres, sans carnet, flux d'ordres ou volume par niveau de prix.
- Le calendrier économique local ne contient que 10 lignes en janvier–mars 2026, dont un événement lié à l'emploi : insuffisant pour une campagne NFP historique.
- La référence ORB conservée a un découpage de 1 500 dates IS jusqu'au 27 février 2024 et 643 dates OOS ensuite. La correction de sortie horaire décrite dans le [mémo institutionnel](../../docs/mnq_orb_strategy_institutional_memo.md) doit accompagner toute comparaison.

Le triage signale **93** fiches compatibles avec les barres OHLCV à première vue, **70** dont les horizons publiés commencent à 1 h ou plus, **16** qui demandent une revue de données tick ou de carnet, **5** qui demandent du volume par prix et **1** qui manque de calendrier historique. Ces classes sont des portes d'entrée pour une revue manuelle, pas une preuve de faisabilité complète.

## Ancienne proposition de panel

Ce panel était une première proposition heuristique. Il est conservé pour la
traçabilité mais ne constitue plus un ordre de travail. La sélection courante
passe d'abord par la taxonomie architecturale, la déduplication et la
[revue logique des moteurs d'alpha](ALPHA_ENGINE_LOGICAL_AUDIT.md).

| Ordre | Fiche | Rôle | Mécanisme à vérifier |
| ---: | --- | --- | --- |
| 1 | [Gap Fill](cards/gap-fill.md) | stratégie | Retour du gap vers la clôture RTH précédente |
| 2 | [VWAP Deviation](cards/vwap-deviation.md) | stratégie | Retour d'un écart intraday au VWAP |
| 3 | [Liquidity Sweep Reversal](cards/liquidity-sweep-reversal.md) | stratégie | Échec d'un franchissement local |
| 4 | [Gap & Go](cards/gap-and-go.md) | stratégie | Continuation après grand gap |
| 5 | [Three Bar Reversal](cards/three-bar-reversal.md) | stratégie | Retournement court sur barres |
| 6 | [Keltner Channel Breakout](cards/keltner-channel-breakout.md) | stratégie | Franchissement de canal |
| 7 | [Volatility Breakout](cards/volatility-breakout.md) | stratégie | Expansion après déclencheur de volatilité |
| 8 | [Range Midpoint Reversion](cards/range-midpoint-reversion.md) | stratégie | Retour vers le milieu d'un range défini causalement |
| 9 | [Pivot Point Bounce](cards/pivot-point-bounce.md) | stratégie | Rebond sur niveau de la séance précédente |
| 10 | [RSI(2) Pullback](cards/rsi-2-pullback.md) | stratégie | Réversion rapide au sein d'une tendance |
| 11 | [Volume Climax Reversal](cards/volume-climax-reversal.md) | stratégie | Épuisement prix-volume |
| 12 | [Narrow Body Candle](cards/narrow-body-candle.md) | stratégie | Expansion après contraction des corps de bougies |
| 13 | [Camarilla Pivot Breakout](cards/camarilla-breakout.md) | stratégie | Breakout de pivot journalier |
| 14 | [Parkinson Range Volatility](cards/parkinson-volatility.md) | variable | Niveau de volatilité high-low |
| 15 | [Kaufman Efficiency](cards/kaufman-efficiency.md) | variable | Persistance directionnelle / tendance |

Le prototype Gap Fill v0 a été préspécifié et calculé sur IS seulement. Il ne passe pas le premier contrôle de rentabilité nette et de complémentarité avec l'ORB; il ne progresse pas vers une recherche de paramètres ou l'OOS sur cette version. Son rapport généré n'a pas été récupéré.

## Règles de passage

1. Une idée reçoit une définition causale, une convention de séance, des coûts et un journal de trades auditables **avant** de calculer sa performance.
2. Une variante proche d'un prototype déjà évalué partage son cluster et sa famille de tests; quatre paires ont pour l'instant seulement un **nom identique** et restent à comparer règle par règle.
3. Les données absentes bloquent le backtest de la logique correspondante. Une approximation OHLCV d'un iceberg, d'une absorption ou d'un vrai POC ne peut pas porter le même nom.
4. Les résultats du 28 février 2024 au 20 mars 2026 ont déjà été consultés dans ce projet, au moins pour l'ORB. Pour une nouvelle décision après cette sélection, ils sont un audit historique, pas un holdout vierge. Une validation réellement nouvelle demandera des données acquises après le 20 mars 2026 ou un protocole prospectif figé.
5. Les comparaisons marginales avec l'ORB utilisent la [référence corrigée à 16:00](../../docs/mnq_orb_strategy_institutional_memo.md). La comparaison prop firm vient après l'existence d'une candidate nette, robuste et distincte.

Reproduction du registre initial : `python strategy_registry/psf_mnq/triage.py`.
Les 16 mécanismes représentatifs disposent maintenant de
[spécifications canoniques figées](FROZEN_BASELINES_REVIEW.md). Aucun nouveau
baseline backtest n'est lancé par cette étape de préparation.
