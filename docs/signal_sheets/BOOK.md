# Book — portefeuille des neuf sleeves

> **Note du 3 novembre 2026.** Ce book décrit la référence à neuf sleeves du 1er novembre 2026. La référence active compte désormais huit sleeves : `S04` est passée en shadow ([`ACTIVE_EIGHT_SLEEVES.md`](../ACTIVE_EIGHT_SLEEVES.md)). Sans S04, la section 5 donne déjà le book restant : `31 661 USD`, Sharpe `1,751`.

## 1. Construction et périmètre

Le book additionne une copie exécutable de chaque sleeve : quatre MNQ, deux MGC et trois MBT, sans pondération fractionnaire. Il correspond à la référence `nine_sleeve_equal_contract_reference_20261101`, dans laquelle les sleeves 2 à 9 exécutent leurs ensembles de vote gelés. Les chiffres de cette section utilisent la fenêtre commune du **2 janvier 2024 au 19 mars 2026** (564 séances) et les PnL quotidiens nets de coûts nominaux. Cette fenêtre rend les comparaisons cohérentes ; elle ne constitue pas un nouvel OOS indépendant.

![Equity curve nette du book](assets/book_equity_curve.png)

Sur cette fenêtre, le book cumule `37 044,22 USD`, avec une volatilité quotidienne de `677,34 USD`, un pire jour de `−1 664,46 USD` et un drawdown maximal clôture-à-clôture de `−6 305,18 USD`. La référence du 4 octobre 2026 affichait `43 225,22 USD` et un drawdown de `−6 014,42 USD` sur la même fenêtre : les anciens points S3–S6 y avaient été choisis, ce qui rendait ces chiffres optimistes. Ces pertes historiques ne bornent pas la perte future : gaps, slippage, liquidité ou défaillance d'exécution peuvent faire dépasser un stop théorique.

## 2. Matrice de corrélation

![Matrice de corrélation des neuf sleeves](assets/book_correlation_matrix.png)

La matrice est la corrélation de Pearson des **PnL quotidiens**, jours sans trade inclus. Les blocs les plus liés sont `S05–S06` (`0,66`, deux stratégies MGC), `S07–S08` (`0,63`, deux modèles MBT de tendance) et `S02–S04` (`0,57`, deux moteurs MNQ directionnels). Une corrélation faible n'empêche pas des pertes simultanées lors d'un choc ; elle décrit uniquement la co-mouvance moyenne observée.

| Code | Sleeve |
|---|---|
| S01 | MNQ ORB + VWAP + ATR |
| S02 | MNQ volatility breakout |
| S03 | MNQ compression breakout |
| S04 | MNQ ATR trend flip |
| S05 | MGC volatility breakout |
| S06 | MGC gap continuation |
| S07 | MBT rolling range breakout |
| S08 | MBT trend pullback MA structure |
| S09 | MBT volatility breakout |

## 3. Book risk — stops et risque prévu

Le **risque prévu** ci-dessous est la distance entre le fill et le niveau d'invalidation, convertie en dollars. C'est une estimation ex ante, pas une perte maximale garantie : un ordre stop peut être exécuté plus loin en cas de gap. Pour les sleeves 2 à 9, les cellules internes n'équivalent pas à plusieurs contrats : seule la cible agrégée `−1/0/+1` est tradée, et elle sort lorsque le vote repasse sous son seuil, ce qui peut arriver avant ou après les stops des cellules.

| Sleeve | Stop-loss structurel | Risque prévu par position exécutable |
|---|---|---|
| S01 | Oui — sous l'opening range | budget visé `≤ 250 USD` par trade ; quantité entière réduite quand le stop s'éloigne |
| S02 | Oui dans chaque cellule — open RTH | environ `k × ATR_n × 2 USD` par cellule, `k∈{0,5;0,6}` ; aucun montant unique pour le vote agrégé |
| S03 | Oui dans chaque cellule — extrême opposé de la setup | `(setup_range + b ticks) × 2 USD` par cellule ; le vote ne porte jamais plus d'un MNQ |
| S04 | Oui dans chaque cellule — fixe à 1 ATR14 | `ATR14 × 2 USD` au fill de chaque cellule ; le vote peut sortir avant le stop |
| S05 | Oui dans chaque cellule — open COMEX | environ `k × ATR_N × 10 USD` par cellule, `k∈{0,9;1,1;1,3}` ; aucun montant unique pour le vote |
| S06 | Oui dans chaque cellule — milieu du gap | `|fill − milieu du gap| × 10 USD` par cellule ; dépend du gap et de la barre de confirmation |
| S07 | Non, pas de stop prix unique | sortie par canal opposé puis hard flatten ; perte intraday non bornée par un SL fixe |
| S08 | Oui dans chaque cellule, pas au niveau agrégé | distance fill–EMA lente variable ; le vote peut modifier la cible avant/après les stops internes |
| S09 | Oui dans chaque cellule — open RTH | environ `k × ATR_N × 0,10 USD` par cellule active ; aucun montant unique pour le vote agrégé |

## 4. Book risk — risque réalisé sur la fenêtre commune

![Profil de risque historique des neuf sleeves](assets/book_risk_profile.png)

| Sleeve | Vol. quotidienne | Pire jour | Max drawdown | Contribution au risque du book |
|---|---:|---:|---:|---:|
| S01 | 145,57 USD | −249,00 USD | −1 081,50 USD | 5,9 % |
| S02 | 290,74 USD | −1 138,00 USD | −3 714,00 USD | 33,1 % |
| S03 | 58,95 USD | −268,50 USD | −940,50 USD | 3,2 % |
| S04 | 279,22 USD | −823,50 USD | −4 212,50 USD | 30,3 % |
| S05 | 123,33 USD | −874,50 USD | −1 622,50 USD | 5,3 % |
| S06 | 149,96 USD | −909,50 USD | −1 399,00 USD | 6,8 % |
| S07 | 73,76 USD | −355,32 USD | −1 285,80 USD | 3,3 % |
| S08 | 105,46 USD | −533,64 USD | −1 449,10 USD | 5,4 % |
| S09 | 99,84 USD | −229,32 USD | −2 068,58 USD | 6,8 % |

La contribution au risque est la contribution marginale à la variance quotidienne du book equal-contract ; elle tient donc compte à la fois de la volatilité propre et des corrélations. `S02` et `S04` concentrent ensemble environ `63,4 %` du risque observé. Le contrôle opérationnel doit aussi agréger les expositions par instrument : quatre contrats MNQ peuvent être simultanément orientés dans le même sens, auxquels peut s'ajouter le MBT directionnel.

## 5. Analyse leave-one-out

Cette analyse retire **une seule sleeve à la fois**, sans réoptimiser les huit restantes. Elle est strictement bornée à la période de recherche commune du **2 janvier 2024 au 19 mars 2026** : toute donnée ajoutée après cette date est exclue par un filtre et un contrôle de bornes dans le script de génération. La ligne pointillée du graphique représente le book complet.

![Analyse leave-one-out du book](assets/book_leave_one_out.png)

| Sleeve retirée | PnL du book restant | Variation de PnL | Sharpe annualisé | Max drawdown | Variation du drawdown |
|---|---:|---:|---:|---:|---:|
| S01 | 30 715,22 USD | −6 329,00 USD | 1,324 | −6 670,64 USD | −365,46 USD |
| S02 | 27 765,22 USD | −9 279,00 USD | 1,595 | −5 704,60 USD | +600,58 USD |
| S03 | 36 451,72 USD | −592,50 USD | 1,559 | −6 543,68 USD | −238,50 USD |
| S04 | 31 661,22 USD | −5 383,00 USD | 1,751 | −5 425,58 USD | +879,60 USD |
| S05 | 35 256,22 USD | −1 788,00 USD | 1,522 | −6 123,18 USD | +182,00 USD |
| S06 | 32 946,22 USD | −4 098,00 USD | 1,433 | −6 416,78 USD | −111,60 USD |
| S07 | 34 689,18 USD | −2 355,04 USD | 1,482 | −6 171,16 USD | +134,02 USD |
| S08 | 32 699,12 USD | −4 345,10 USD | 1,419 | −6 368,38 USD | −63,20 USD |
| S09 | 34 169,64 USD | −2 874,58 USD | 1,509 | −6 279,24 USD | +25,94 USD |

Le book complet affiche un PnL de `37 044,22 USD`, un Sharpe annualisé de `1,539` et un drawdown maximal de `−6 305,18 USD`. Toutes les sleeves ont une contribution nette positive sur cette fenêtre : en retirer une diminue donc toujours le PnL final. `S04` puis `S02` sont les principaux compromis rendement/risque : leur retrait réduit le PnL, mais améliore le Sharpe et le drawdown, ce qui confirme leur forte contribution au risque. À l'inverse, retirer `S01`, `S03`, `S06` ou `S08` dégrade le drawdown malgré la baisse d'exposition ; ces sleeves apportent une diversification utile sur cette fenêtre. `S03` a la plus faible contribution au PnL, mais son retrait aggrave le drawdown. Le gain de `S09` sur cette fenêtre provient surtout de 2024 et du début 2025 : son vote est nul sur son legacy OOS (fiche 09).

Une variation de drawdown positive signifie que le drawdown devient moins sévère ; une variation négative signifie qu'il se dégrade. Cette analyse est descriptive et rétrospective : elle mesure la dépendance historique du book à chaque sleeve, sans valider une nouvelle allocation.

## 6. Garde-fous de production

- Calculer en temps réel le risque jusqu'au stop de chaque position, avec la quantité réellement remplie et les stops effectivement présents chez le broker.
- Traiter `S07` et les sleeves en vote comme des risques non bornés par un montant fixe tant qu'aucun stop portefeuille distinct n'est imposé : la cible agrégée ne porte pas les stops des cellules chez le broker.
- Suivre les expositions brutes par instrument, les pertes simultanées et le drawdown du book, pas seulement chaque sleeve isolément.
- Ne jamais utiliser le drawdown historique comme limite de perte garantie ; ajouter des limites de journée et de book indépendantes si le mandat d'exécution l'exige.

Sources : [PnL quotidien canonique des neuf sleeves](../../research/locked_nine_sleeve_portfolio_20261101/daily_pnl.csv), [configuration verrouillée](../../research/locked_nine_sleeve_portfolio_20261101/locked_nine_sleeve_config.json), [comparaison avec la référence remplacée](../../research/locked_nine_sleeve_portfolio_20261101/book_risk_comparison.csv).
