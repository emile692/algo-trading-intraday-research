# Campagne ciblée 6E/ZN — décision finale

Date de gel et d'exécution : 2026-09-26. Période complète : 2015-01-02 au
2026-03-19. IS : 2019-05-06–2023-12-31. OOS : 2024-01-01–2026-03-19.
2026 est donc une année partielle. Capital conventionnel : 50 000 USD ; une
unité par trade ; Sharpe sur PnL quotidien avec jours ouvrés sans trade à zéro.

## Verdict

**STOP définitif de la branche 6E/ZN.** Aucun des six frameworks ne franchit
les portes standalone préspecifiées. Le test portefeuille n'est donc pas
déclenché et aucune optimisation supplémentaire n'est autorisée. La prochaine
campagne est MCL/CL avec les 16 mécanismes frozen.

## Résultats canoniques nets (un tick de slippage par côté)

| Framework | IS trades | IS PnL | IS Sharpe | OOS trades | OOS PnL | OOS Sharpe | OOS PF | OOS max DD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eur_benchmark_fix_reversal | 2 408 | 3 686 | 0,10 | 1 143 | -6 126 | -0,39 | 0,95 | -18 136 |
| regional_working_hours_fx_flow | 2 408 | 3 843 | 0,11 | 1 143 | -9 189 | -0,59 | 0,93 | -19 130 |
| scheduled_macro_acceptance | 84 | -1 023 | -0,20 | 48 | -3 189 | -1,22 | 0,51 | -4 079 |
| ten_year_auction_concession_reversal | 172 | -5 524 | -1,12 | 82 | -299 | -0,11 | 0,94 | -2 367 |
| scheduled_treasury_information_assimilation | 132 | -3 346 | -0,72 | 67 | -1 824 | -0,84 | 0,68 | -1 914 |
| treasury_day_night_segmentation | 1 163 | -64 048 | -2,32 | 551 | -22 315 | -2,05 | 0,70 | -23 587 |

| Framework | Full trades | Full PnL | Full Sharpe | Full PF | Full max DD |
|---|---:|---:|---:|---:|---:|
| eur_benchmark_fix_reversal | 5 789 | 40 940 | 0,44 | 1,06 | -21 490 |
| regional_working_hours_fx_flow | 5 789 | 31 853 | 0,34 | 1,04 | -22 009 |
| scheduled_macro_acceptance | 220 | -5 975 | -0,42 | 0,78 | -8 928 |
| ten_year_auction_concession_reversal | 410 | -10 338 | -0,91 | 0,67 | -11 773 |
| scheduled_treasury_information_assimilation | 325 | -6 094 | -0,58 | 0,76 | -8 375 |
| treasury_day_night_segmentation | 2 799 | -113 748 | -1,96 | 0,71 | -116 079 |

Le vrai post-échantillon de l'étude d'adjudications (2024-08-01+) reste
négatif : -494 USD, Sharpe -0,22, PF 0,89 sur 60 jambes. Les trois fenêtres
150/180/210 minutes sont toutes négatives sur cette période.

## Stabilité et interprétation économique

- Les 18 combinaisons framework/perturbation sont négatives OOS. La stabilité
  locale OOS vaut donc 0/3 pour chacun des six frameworks.
- Les deux règles FX sont positives sur l'historique complet seulement à cause
  des premières années. Elles deviennent négatives OOS et à deux ticks. À zéro
  slippage, leur OOS redevient positif : l'effet net n'est pas exploitable de
  façon robuste avec le modèle de coûts du projet.
- La jambe short européenne pré-fix reste légèrement positive OOS (+1 473 USD,
  Sharpe 0,13, PF 1,02), mais la jambe de reversal après le fix est fortement
  négative (-7 599 USD, Sharpe -0,75). Cela ne valide pas le framework de
  reversal préspecifié et ne justifie pas d'extraire après coup une nouvelle
  stratégie.
- L'assimilation Treasury est négative dans son ensemble. Les sous-échantillons
  FOMC et ISM sont faiblement positifs OOS mais négatifs IS, avec seulement 13
  trades chacun : aucun signal persistant.
- Pour les adjudications, la concession short et le reversal long sont tous
  deux négatifs sur l'échantillon complet. Le résultat post-juillet 2024 ne
  confirme pas le mécanisme net de coûts.
- La segmentation day/night ZN est négative chaque année, avec un rolling
  Sharpe 252 jours médian de -2,01. Ce n'est pas une instabilité marginale mais
  un signe opposé et persistant sous cette règle futures intraday.

## Coûts et rolling Sharpe

À deux ticks par côté, les PnL OOS canoniques sont respectivement -20 414,
-23 476, -3 789, -2 861, -3 918 et -39 534 USD dans l'ordre du tableau.
Les derniers rolling Sharpes 252 jours sont -0,48, -0,70, -1,30, -0,52,
-1,35 et -2,34. Les historiques complets par année, rolling Sharpe et niveau de
slippage sont conservés dans les CSV du dossier.

## Données et contrôles

Les calendriers ont été gelés avant le passage valide : 134 NFP, 134 CPI, 90
FOMC, 135 dates ISM et 205 adjudications/réouvertures 10Y. Les réouvertures sont
identifiées via `original_security_term=10-Year`; les exclure aurait laissé
seulement 68 émissions et constitué un calendrier incomplet. Les horaires
Europe/Berlin, Europe/London et America/New_York utilisent leurs zones civiles,
y compris les semaines de décalage DST.

Trois tests automatisés couvrent les conversions DST, les quatre côtés du
slippage adverse et le coût complet d'un aller-retour. Ils passent tous.
