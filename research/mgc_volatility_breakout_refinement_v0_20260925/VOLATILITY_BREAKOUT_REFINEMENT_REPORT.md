# MGC — Volatility Breakout, raffinement local IS / Validation

## Protocole

- IS : `2019-05-06 → 2023-12-31`.
- Validation : `2024-01-01 → 2026-03-19`.
- Grille : `[3, 5, 7, 10, 14, 20, 30, 40, 50]` × `[0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]` (81 cellules).
- Signaux, sorties, coûts, fills, session, taille et money management inchangés.
- Ce raffinement a été défini après lecture de la campagne large : la Validation est réutilisée et ne constitue donc plus un OOS vierge.
- SHA-256 de la grille : `9adb102db9706e01523ca34db827ea4bf692fe3e27dfa6756dcfda63a23d60b5`.

## Résultat descriptif

- Cellules positives simultanément : `37/81`.
- Plus grande composante positive contiguë : `35` cellules.
- Meilleure cellule selon le minimum des Sharpes IS/Validation : `daily_atr_period=50`, `entry_atr_multiple=1.3`.
- Sharpe IS / Validation : `0.487` / `0.472`.
- Trades IS / Validation : `79` / `42`.

Le classement complet utilise d'abord `min(Sharpe IS, Sharpe Validation)`, puis la moyenne des deux Sharpes. Il est descriptif et ne transforme pas cette Validation réutilisée en nouveau test hors-échantillon.
