# Gap Fill MNQ — protocole v0, figé avant calcul

Source d'idée : [fiche PineScriptForge](../cards/gap-fill.md). Cette spécification locale tranche les ambiguïtés de la description publique; elle ne prétend pas reproduire les chiffres du site.

## Hypothèse

Un petit écart entre la clôture RTH précédente et l'ouverture RTH suivante tend à revenir vers la clôture précédente avant midi. L'intérêt marginal face à l'ORB sera évalué sur les journées communes.

## Données et échantillon

- MNQ continu 1 minute : `data/processed/parquet/MNQ_c_0_1m_20260321_094501.parquet`, horodatage de début de barre en `America/New_York`.
- Étude initiale : sessions jusqu'au **27 février 2024** inclus, le découpage IS de la référence ORB. La période du 28 février 2024 au 20 mars 2026 n'entre pas dans la décision de cette version.
- Séance RTH : barres de 09:30 à 15:59. On exige les barres exactes de 09:30, 09:31 et une clôture précédente à 15:59, ainsi qu'au moins 300 barres RTH dans chaque séance liée.
- Une transition d'`instrument_id` entre la clôture précédente et l'ouverture actuelle exclut la séance. L'ATR utilise uniquement des séances dont la transition de contrat est valide.

## Signal et exécution

1. `gap = open_RTH[t] - close_RTH[t-1]`. Ignorer un gap inférieur à un tick MNQ (0,25 point) ou supérieur ou égal à l'ATR 14 séances connu la veille.
2. ATR 14 : moyenne simple des 14 derniers *true ranges* RTH valides, décalée d'une séance. Le true range utilise le high/low RTH et la clôture RTH précédente. Aucune donnée de la séance actuelle ne contribue à l'ATR d'entrée.
3. Gap positif : direction short; gap négatif : direction long. Objectif = clôture RTH précédente. Stop = ouverture RTH actuelle prolongée de 50 % du gap dans sa direction.
4. Lire la barre 09:30 jusqu'à sa clôture. Si elle touche déjà objectif ou stop, ignorer la journée. Entrer à l'ouverture exacte de 09:31 si ce prix reste strictement entre objectif et stop. Au plus un trade par jour.
5. Examiner les barres de 09:31 à 11:59. Si stop et objectif sont touchés dans la même barre, attribuer le stop en premier. Un stop franchi à l'ouverture est exécuté au moins à cette ouverture défavorable; un objectif franchi à l'ouverture reste exécuté au prix objectif. Sans sortie précédente, sortir à l'ouverture de 12:00. Si cette barre manque, utiliser la dernière clôture disponible avant midi.
6. Une seule position MNQ pour cette baseline. Commission : 1,25 $ par côté. Slippage : 1 tick défavorable par côté. PnL en dollars : 2 $ par point et par contrat.

## Lecture prévue

Publier nombre de séances éligibles, motifs d'exclusion, trades, PnL net, espérance par trade, profit factor, Sharpe sur toutes les dates du découpage IS et drawdown. Comparer les PnL quotidiens avec l'ORB corrigé sur les mêmes dates, en gardant visibles les journées sans trade. Une analyse de sensibilité ou un changement de seuil constituera une **nouvelle version** de l'expérience. Un résultat IS favorable restera une hypothèse de recherche; il ne validera ni l'OOS déjà consulté ni une exécution prop firm.
