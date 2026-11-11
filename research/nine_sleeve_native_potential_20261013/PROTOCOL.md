# Protocole gelé avant calcul

Campagne diagnostique isolée. Aucun fichier source, signal, paramètre alpha, stop, target, horaire ou book actif n'est modifié.

- Unité native : une copie exécutable, soit un microcontrat par trade. S01 est reconstruite à un micro à partir de son ledger historiquement sizé à 250 USD.
- S02 : sortie effective par time-stop 240 minutes. Le niveau RTH open publié n'est pas traité comme stop exécuté ; seules les quantités fixes sont interprétables.
- S07–S09 : chaque sleeve est l'ensemble de votes causal discrétisé exporté, et non ses cellules comme ordres séparés. Sans stop initial unique, seules les quantités fixes sont interprétables.
- Quantités fixes : 1, 2, 3, 4 micros. Budgets : 100, 150, 200, 250, 350, 500, 750 USD uniquement pour S01, S03–S06.
- Sizing : `min(50, floor(budget / (distance causale au stop * point value + commission RT + allowance d'exécution nominale RT)))`. Le plafond 50 est la limite commune la plus restrictive des comptes Topstep testés. Quantité zéro = signal ignoré. Quantité fixée à l'entrée, sans pyramiding ni rebalancement.
- Prix prévu : trigger gelé pour les stop-orders ; dernière clôture minute observable pour les market-orders ; distance ATR gelée de la dernière barre achevée pour S04. Le fill futur n'augmente jamais la quantité. L'écart fill/plan et les dépassements sont audités.
- Coûts : conventions gelées, puis stress de +1 et +2 ticks par côté. Les PnL sont recalculés trade par trade.
- Calendrier : toutes les séances du ledger daily natif, jours sans trade inclus dans le Sharpe.
- Incertitude : moving-block bootstrap apparié, blocs 5/10/20 séances. Les fenêtres Topstep sont étiquetées `UNRESOLVED` au terme de 35/60/120 séances ; il n'existe pas d'expiration contractuelle dans les règles vérifiées.
- Les périodes déjà inspectées lors de la sélection ne sont jamais qualifiées de nouvel OOS indépendant.
