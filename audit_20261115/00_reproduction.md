# Phase 0 — Reproduction

*Audit du 2026-11-15. Référence auditée : `nine_sleeve_equal_contract_reference_20261101`. Script : [phase0_reproduction.py](phase0_reproduction.py) et [phase0_engine.py](phase0_engine.py). Table brute : [outputs/00_reproduction.csv](outputs/00_reproduction.csv).*

## Verdict

**65 contrôles sur 67 passent. 2 échouent.** L'audit s'est arrêté ici le 2026-11-15, conformément à la règle 5. Il a repris après les décisions de l'utilisateur, en fin de document.

Les PnL, les nombres de trades, le book, le leave-one-out et le netting se reproduisent au centime. Les échecs ne viennent pas des données ni du moteur. Ils viennent de deux incohérences de définition dans la documentation : décision 0.1 (la version de S01) et décision 0.2 (la convention de Sharpe de S02).

## Conventions

- Sharpe : √252 × moyenne / écart-type (ddof = 1) du PnL quotidien net en USD, jours sans trade inclus, sur le calendrier du fichier source de chaque sleeve. Les Sharpe publiés étant arrondis à 3 décimales, c'est la valeur arrondie qui est comparée, avec la tolérance de 0,005.
- Coûts nominaux du lock : 1,25 USD de commission par côté (MNQ, MGC), 1,41 USD (MBT), 1 tick de slippage par côté.
- Fenêtres : MNQ/MGC 2019-05-06 → 2026-03-19 (IS ≤ 2023-12-31), MBT 2021-05-03 → 2026-10-02 (IS ≤ 2025-05-28), S01 2019-05-05 → 2026-03-20 (IS ≤ 2024-02-27), book 2024-01-02 → 2026-03-19.
- Niveau A : métriques recalculées depuis les fichiers gelés. Niveau B : ré-exécution des sleeves depuis les barres 1 minute avec les moteurs du repo, comparée trade par trade. Niveau C : moteur de netting `src/portfolio/aggregator.py`, barres coupées au 2026-03-19.

## Table attendu / obtenu / écart

| Élément | Métrique | Attendu | Obtenu | Écart | OK |
|---|---|---:|---:|---:|---|
| S01 full | trades | 701 | 701 | 0 | oui |
| S01 full | pnl | 19 416,50 | 19 416,50 | 0,00 | oui |
| S01 full | sharpe | 1,1560 | 1,1555 | −0,0005 | oui |
| S01 full | pf | 1,3390 | 1,3389 | −0,0001 | oui |
| S01 full | maxdd | −2 148,50 | −2 148,50 | 0,00 | oui |
| S02 full | trades | 1085 | 1085 | 0 | oui |
| S02 full | pnl | 20 259,00 | 20 259,00 | 0,00 | oui |
| S02 full | sharpe | 0,8730 | 0,8476 | −0,0254 | **NON** |
| S03 full | trades | 329 | 329 | 0 | oui |
| S03 full | pnl | 2 255,50 | 2 255,50 | 0,00 | oui |
| S03 full | sharpe | 0,4320 | 0,4322 | 0,0002 | oui |
| S04 full | trades | 2052 | 2052 | 0 | oui |
| S04 full | pnl | 10 075,00 | 10 075,00 | 0,00 | oui |
| S04 full | sharpe | 0,4240 | 0,4242 | 0,0002 | oui |
| S05 full | trades | 197 | 197 | 0 | oui |
| S05 full | pnl | 3 135,50 | 3 135,50 | 0,00 | oui |
| S05 full | sharpe | 0,3700 | 0,3701 | 0,0001 | oui |
| S06 full | trades | 231 | 231 | 0 | oui |
| S06 full | pnl | 7 030,50 | 7 030,50 | 0,00 | oui |
| S06 full | sharpe | 0,6770 | 0,6769 | −0,0001 | oui |
| S07 full | trades | 461 | 461 | 0 | oui |
| S07 full | pnl | 2 082,48 | 2 082,48 | 0,00 | oui |
| S07 full | sharpe | 0,4730 | 0,4727 | −0,0003 | oui |
| S08 full | trades | 1142 | 1142 | 0 | oui |
| S08 full | pnl | 3 900,06 | 3 900,06 | 0,00 | oui |
| S08 full | sharpe | 0,6420 | 0,6421 | 0,0001 | oui |
| S09 full | trades | 668 | 668 | 0 | oui |
| S09 full | pnl | 4 916,24 | 4 916,24 | 0,00 | oui |
| S09 full | sharpe | 0,8740 | 0,8741 | 0,0001 | oui |
| S09 val | pnl | −73,22 | −73,22 | 0,00 | oui |
| S09 val | sharpe | −0,0400 | −0,0403 | −0,0003 | oui |
| Book 9 (lock CSV) | sessions | 564 | 564 | 0 | oui |
| Book 9 (lock CSV) | pnl | 37 044,22 | 37 044,22 | 0,00 | oui |
| Book 9 (lock CSV) | sharpe | 1,5390 | 1,5393 | 0,0003 | oui |
| Book 9 (lock CSV) | maxdd | −6 305,18 | −6 305,18 | 0,00 | oui |
| Book 9 (lock CSV) | worst | −1 664,46 | −1 664,46 | 0,00 | oui |
| Book 9 reconstruit depuis les sources | écart quotidien absolu max vs lock | 0,00 | 0,00 | 0,00 | oui |
| S01 lock vs S01 signal_sheet_01_atr18 | écart quotidien absolu max | 0,00 | 474,00 | 474,00 | **NON** |
| LOO sans S01 | pnl | 30 715,22 | 30 715,22 | 0,00 | oui |
| LOO sans S01 | sharpe | 1,3240 | 1,3241 | 0,0001 | oui |
| LOO sans S01 | maxdd | −6 670,64 | −6 670,64 | 0,00 | oui |
| LOO sans S02 | pnl | 27 765,22 | 27 765,22 | 0,00 | oui |
| LOO sans S02 | sharpe | 1,5950 | 1,5953 | 0,0003 | oui |
| LOO sans S02 | maxdd | −5 704,60 | −5 704,60 | 0,00 | oui |
| LOO sans S03 | pnl | 36 451,72 | 36 451,72 | 0,00 | oui |
| LOO sans S03 | sharpe | 1,5590 | 1,5590 | −0,0000 | oui |
| LOO sans S03 | maxdd | −6 543,68 | −6 543,68 | 0,00 | oui |
| LOO sans S04 | pnl | 31 661,22 | 31 661,22 | 0,00 | oui |
| LOO sans S04 | sharpe | 1,7510 | 1,7508 | −0,0002 | oui |
| LOO sans S04 | maxdd | −5 425,58 | −5 425,58 | 0,00 | oui |
| LOO sans S05 | pnl | 35 256,22 | 35 256,22 | 0,00 | oui |
| LOO sans S05 | sharpe | 1,5220 | 1,5218 | −0,0002 | oui |
| LOO sans S05 | maxdd | −6 123,18 | −6 123,18 | 0,00 | oui |
| LOO sans S06 | pnl | 32 946,22 | 32 946,22 | 0,00 | oui |
| LOO sans S06 | sharpe | 1,4330 | 1,4331 | 0,0001 | oui |
| LOO sans S06 | maxdd | −6 416,78 | −6 416,78 | 0,00 | oui |
| LOO sans S07 | pnl | 34 689,18 | 34 689,18 | 0,00 | oui |
| LOO sans S07 | sharpe | 1,4820 | 1,4824 | 0,0004 | oui |
| LOO sans S07 | maxdd | −6 171,16 | −6 171,16 | 0,00 | oui |
| LOO sans S08 | pnl | 32 699,12 | 32 699,12 | 0,00 | oui |
| LOO sans S08 | sharpe | 1,4190 | 1,4191 | 0,0001 | oui |
| LOO sans S08 | maxdd | −6 368,38 | −6 368,38 | 0,00 | oui |
| LOO sans S09 | pnl | 34 169,64 | 34 169,64 | 0,00 | oui |
| LOO sans S09 | sharpe | 1,5090 | 1,5091 | 0,0001 | oui |
| LOO sans S09 | maxdd | −6 279,24 | −6 279,24 | 0,00 | oui |
| Netting 2024-01→2026-03 | PnL séparé (somme des trades) | 31 152,00 | 31 151,96 | −0,04 | oui |
| Netting 2024-01→2026-03 | PnL agrégé (moteur de netting, cap 5 lots) | 31 180,00 | 31 179,96 | −0,04 | oui |

## Écart 1 — le S01 du book n'est pas le S01 de la fiche 01 (bloquant)

Les chiffres de la fiche 01 (701 trades, 19 416,50 USD, Sharpe 1,156) se reproduisent. Mais le book n'utilise pas cette sleeve. Trois ledgers S01 coexistent dans le repo :

| Source | Définition | Trades | PnL total | Utilisée par |
|---|---|---:|---:|---|
| `research/signal_sheet_01_atr18/trades.csv` | vote de 18 modèles : ATR {14, 16, 18} × q_low {20, 25, 30} × q_high {80, 85} | 701 | 19 416,50 | fiche 01 |
| `export/mnq_orb_vwap_atr_retained/trades.csv` | `baseline_fixed` : ATR 14, q_low {20, 25, 30} × q_high {90, 95} | 823 | 18 537,50 | étude de netting, règle de netting du 15/11 |
| `research/portfolio_integration_phase1/sleeve_trades/mnq_orb_vwap_atr.csv` | extrait 2024+ de `baseline_fixed` | 261 | 6 329,00 | locks du 01/11 et du 03/11, BOOK.md sections 1–5 |

Sur 2024-01-02 → 2026-03-19, le S01 du lock fait 6 329 USD sur 261 trades. Celui de la fiche fait 6 035 USD sur 208 trades. Les 208 trades de la fiche sont tous inclus dans les 261 du lock, qui en compte 53 de plus. L'écart quotidien maximal atteint 474,00 USD. Le forward replay (`frozen_vote_forward_replay_20261101`) et la règle de netting reprennent eux aussi `baseline_fixed`.

Effet sur le book, fenêtre commune 2024-01-02 → 2026-03-19, 564 séances :

| Version de S01 | Sleeves | PnL | Sharpe | maxDD |
|---|---:|---:|---:|---:|
| S01 = `baseline_fixed` (lock, book, netting) | 9 | 37 044,22 | 1,539 | −6 305,18 |
| S01 = `baseline_fixed` (lock, book, netting) | 8 | 31 661,22 | 1,751 | −5 425,58 |
| S01 = ensemble ATR 18 (fiche 01) | 9 | 36 750,22 | 1,541 | −6 350,78 |
| S01 = ensemble ATR 18 (fiche 01) | 8 | 31 367,22 | 1,754 | −5 626,08 |

L'effet sur le book est faible : −294 USD et Sharpe quasi inchangé. Le problème est ailleurs : la définition exécutable de S01 est ambiguë. La fiche, qui sert de spécification de réimplémentation, décrit un signal que ni le lock ni la production ne tradent.

## Écart 2 — le Sharpe publié de S02 n'est pas un Sharpe en USD

Le `summary.json` de S02 a été produit par `compute_metrics` du moteur de base. Ce moteur calcule le Sharpe sur le rendement quotidien d'un compte de 50 000 USD qui capitalise (`daily_pnl / equity_{t−1}`). S03 à S09 utilisent `metric_block`, qui calcule le Sharpe sur le PnL en USD. Les deux conventions divergent dès que l'equity s'éloigne de 50 000 USD.

| S02 | PnL | Sharpe publié (rendements) | Sharpe en USD (convention de l'audit) |
|---|---:|---:|---:|
| IS 2019-05-06 → 2023-12-31 | 10 980,00 | 0,831 | 0,859 |
| Validation 2024-01-01 → 2026-03-19 | 9 279,00 | 0,971 | 0,899 |
| Total | 20 259,00 | 0,873 | 0,848 |

Les Sharpe en rendements de S03 à S09 seraient aussi plus élevés (de +0,006 à +0,037). Seul S02 publie cette convention-là. Le PnL et les trades de S02 sont exacts.

## Points secondaires, non bloquants

- **Calendrier de S01.** `signal_sheet_01_atr18/daily_results.csv` contient 2 143 lignes, dont 355 dimanches sans séance RTH. Le Sharpe publié de 1,156 est calculé avec ces zéros. Sur le calendrier MNQ de S02 (1 747 séances), le même PnL donne un Sharpe de 1,281. La sleeve n'a aucun trade un dimanche : seuls le dénominateur et la convention changent.
- **Hash périmé.** `signal_sheet_02_cluster/summary.json` référence un hash du manifeste `volatility_breakout_42.json` (`a1cfe825…`) différent du hash actuel (`e3e912ad…`), qui est celui du lock. Le manifeste a été réécrit après l'exécution, quand le bloc `execution_results` y a été ajouté. Le lock pointe vers la bonne version.
- **Netting.** 31 151,96 USD en séparé et 31 179,96 USD en agrégé (cap de 5 lots, jamais atteint : la position maximale théorique du book est de 3,8 lots : 0,6 MNQ + 0,2 MGC + 3 MBT). Les deux chiffres sont reproduits à 0,04 USD près, ce qui n'est qu'un arrondi de la publication.

## Ré-exécution depuis les barres brutes (niveau B)

Non exécuté dans ce run (`--no-engine`).

## Décisions prises le 2026-11-15

**0.1 — S01 de référence : l'ensemble ATR 18 de la fiche 01** (ATR {14, 16, 18} × q_low {20, 25, 30} × q_high {80, 85}, 701 trades), décision de l'utilisateur. Conséquences :

- toutes les phases suivantes utilisent `research/signal_sheet_01_atr18` pour S01 ;
- **écart à corriger hors audit** : les locks `nine_sleeve_…_20261101` et `eight_sleeve_…_20261103`, la règle de netting `netting_rule_20261115.json`, l'étude de netting et le forward replay contiennent `baseline_fixed`. Leurs chiffres de book ne décrivent donc pas le book de référence. La correction passe par un nouveau lock versionné ; l'audit n'y touche pas (règle 2) ;
- les chiffres « attendus » du book et du leave-one-out de la table ci-dessus restent ceux du lock publié. La phase 1 donne les chiffres du book de référence.

**0.2 — Convention de Sharpe : PnL quotidien en USD pour toutes les sleeves** (règle 4 de la mission). Le 0,873 de la fiche 02 est un Sharpe en rendements ; en USD, il vaut 0,848.

**0.3 — Nouveau lock versionné, sur instruction explicite de l'utilisateur** (« fais le nouveau lock versionné », 2026-11-15). C'est une dérogation à la règle 2 de la mission, demandée par l'utilisateur. Contenu :

- `research/locked_eight_sleeve_portfolio_20261115/` (`eight_sleeve_equal_contract_reference_20261115`) : S01 = ensemble ATR 18 et calendrier union. Rien d'autre ne change ;
- `strategy_registry/execution/netting_rule_20261115_r2.json` : même logique de netting, appliquée au nouveau lock ;
- le registre `forward_monitoring`, `ACTIVE_EIGHT_SLEEVES.md`, des notes datées dans BOOK.md et la fiche 01, et `SUPERSEDED.md` dans le lock du 03/11 ;
- les tests `tests/test_locked_eight_sleeve_portfolio_20261115.py` ; le test de registre de l'ancien lock est remplacé par un test « référence remplacée » ;
- le lock ne calcule pas S01 ATR 18 sur 2026-03-20 → 2026-10-02, fenêtre réservée par l'option (O) du pré-enregistrement.
