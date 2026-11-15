# Phase 5 — Règles d'arrêt

*Historique gelé de chaque sleeve, sur le calendrier de son instrument : MNQ/MGC du 2019-05-06 au 2026-03-19, MBT du 2021-05-03 au 2026-10-02. Book 8 sur le calendrier union, du 2021-05-03 au 2026-03-19. S01 = ATR 18. Bootstrap stationnaire, bloc moyen de 20 séances : 10 000 chemins de 500 séances pour les seuils, 4 000 par scénario pour les délais ; seed 20261120. Script : [phase5_stop_rules.py](phase5_stop_rules.py). Tables : [05_detection.csv](outputs/05_detection.csv), [05_historical_alarms.csv](outputs/05_historical_alarms.csv), [05_distributions.csv](outputs/05_distributions.csv), [05_reinstatement.csv](outputs/05_reinstatement.csv).*

## Point de départ : une règle existe déjà

Contrairement à ce que dit la mission, une règle d'arrêt est enregistrée depuis le 2026-11-03 (`ACTIVE_EIGHT_SLEEVES.md`). C'est **R0** : la somme des 126 dernières séances doit rester au-dessus du minimum de cette même somme sur l'historique gelé. Par construction, R0 ne s'est jamais déclenchée sur l'historique. Ses fausses alarmes et ses délais ne se mesurent donc qu'en simulation.

## Règles comparées

Chaque règle est évaluée à chaque séance, depuis le début du suivi (h séances écoulées).

- **R0**, la règle en vigueur : PnL sur 126 séances inférieur au minimum historique de cette somme.
- **R1** : drawdown maximal depuis le début inférieur au p5(h) du maxDD bootstrap, soit une sévérité au-delà du p95.
- **R2** : même chose avec le p1(h), soit une sévérité au-delà du p99.
- **R3** : PnL roulant sur 120 séances inférieur au p5 du PnL bootstrap à 120 séances.

## Seuils

| Série | Historique | R0 : seuil 126 s. | R1 : maxDD p5 à 250 s. | R2 : maxDD p1 à 250 s. | R3 : p5 PnL 120 s. |
|---|---|---:|---:|---:|---:|
| S01 | 2019-05-06 → 2026-03-19 (1747) | −890,50 | −2 210,57 | −2 806,05 | −639,52 |
| S02 | 2019-05-06 → 2026-03-19 (1747) | −4 019,00 | −5 070,57 | −6 595,51 | −2 177,57 |
| S03 | 2019-05-06 → 2026-03-19 (1747) | −975,00 | −1 128,03 | −1 473,03 | −591,05 |
| S04 | 2019-05-06 → 2026-03-19 (1747) | −2 974,00 | −4 898,02 | −6 281,53 | −2 497,03 |
| S05 | 2019-05-06 → 2026-03-19 (1775) | −1 021,00 | −1 868,05 | −2 591,51 | −982,02 |
| S06 | 2019-05-06 → 2026-03-19 (1775) | −1 006,50 | −1 756,05 | −2 308,12 | −924,02 |
| S07 | 2021-05-03 → 2026-10-02 (1085) | −840,64 | −1 642,53 | −2 144,58 | −809,70 |
| S08 | 2021-05-03 → 2026-10-02 (1085) | −1 284,10 | −2 368,60 | −3 009,66 | −1 149,57 |
| S09 | 2021-05-03 → 2026-10-02 (1085) | −1 479,42 | −1 751,25 | −2 239,69 | −717,72 |
| Book 8 | 2021-05-03 → 2026-03-19 (1261) | n/d | −6 397,54 | −8 285,41 | −1 067,77 |

Les distributions à 20, 60, 120 et 250 séances (p5, p50 du PnL ; p50, p5, p1 du maxDD) sont dans [05_distributions.csv](outputs/05_distributions.csv).

## Compromis fausses alarmes / délai — synthèse (médiane des 9 sleeves)

- **Fausse alarme** : probabilité qu'une sleeve dont le vrai Sharpe est celui de son historique soit coupée dans ses 250 premières séances de suivi.
- **Détection** : même probabilité quand le PnL quotidien est décalé pour un Sharpe vrai de 0, puis de −0,5 (volatilité inchangée).
- **Délai** : nombre de séances avant la première alarme, parmi les chemins qui en déclenchent une sur 500 séances.

| Règle | Fausse alarme médiane ≤ 250 s. | Détection médiane ≤ 250 s., SR 0 | Détection médiane ≤ 250 s., SR −0,5 | Délai médian, SR −0,5 |
|---|---:|---:|---:|---:|
| R0 | 17,0 % | 30,7 % | 54,0 % | 204 séances |
| R1 | 28,4 % | 40,4 % | 53,5 % | 81 séances |
| R2 | 7,3 % | 14,4 % | 23,2 % | 178 séances |
| R3 | 24,8 % | 46,2 % | 66,3 % | 174 séances |
| R0∪R2 | 21,1 % | 35,8 % | 58,6 % | 182 séances |

## Détail par série

| Série | Règle | Fausse alarme ≤ 250 s. (sleeve saine) | Détection ≤ 250 s., SR vrai 0 | délai médian | Détection ≤ 250 s., SR vrai −0,5 | délai médian |
|---|---|---:|---:|---:|---:|---:|
| S01 | R0 | 20,5 % | 69,6 % | 171 | 84,7 % | 135 |
| S01 | R1 | 33,4 % | 70,8 % | 53 | 85,5 % | 51 |
| S01 | R2 | 9,7 % | 38,0 % | 161 | 58,7 % | 144 |
| S01 | R3 | 31,1 % | 79,5 % | 147 | 90,9 % | 122 |
| S01 | R0∪R2 | 25,9 % | 73,9 % | 150 | 86,8 % | 126 |
| S02 | R0 | 5,7 % | 18,0 % | 263 | 30,0 % | 250 |
| S02 | R1 | 26,9 % | 39,6 % | 82 | 51,9 % | 122 |
| S02 | R2 | 7,3 % | 13,7 % | 170 | 23,2 % | 223 |
| S02 | R3 | 23,2 % | 53,2 % | 202 | 71,2 % | 162 |
| S02 | R0∪R2 | 10,2 % | 22,4 % | 236 | 34,8 % | 228 |
| S03 | R0 | 4,0 % | 8,5 % | 284 | 19,2 % | 266 |
| S03 | R1 | 27,6 % | 36,9 % | 39 | 51,5 % | 81 |
| S03 | R2 | 7,1 % | 11,0 % | 78 | 18,9 % | 199 |
| S03 | R3 | 21,8 % | 36,3 % | 230 | 57,6 % | 190 |
| S03 | R0∪R2 | 8,9 % | 14,7 % | 216 | 25,4 % | 238 |
| S04 | R0 | 16,8 % | 30,7 % | 252 | 54,0 % | 204 |
| S04 | R1 | 31,6 % | 44,5 % | 41 | 62,1 % | 70 |
| S04 | R2 | 8,2 % | 15,2 % | 94 | 27,9 % | 176 |
| S04 | R3 | 26,3 % | 43,5 % | 225 | 66,3 % | 174 |
| S04 | R0∪R2 | 21,1 % | 35,8 % | 225 | 58,6 % | 182 |
| S05 | R0 | 17,0 % | 25,1 % | 249 | 41,5 % | 215 |
| S05 | R1 | 25,8 % | 30,7 % | 34 | 40,8 % | 91 |
| S05 | R2 | 6,9 % | 8,9 % | 65 | 13,9 % | 178 |
| S05 | R3 | 18,3 % | 26,6 % | 247 | 43,2 % | 209 |
| S05 | R0∪R2 | 19,9 % | 28,1 % | 234 | 43,6 % | 204 |
| S06 | R0 | 18,6 % | 44,0 % | 218 | 67,7 % | 172 |
| S06 | R1 | 28,4 % | 44,2 % | 90 | 65,3 % | 111 |
| S06 | R2 | 7,2 % | 16,3 % | 192 | 32,0 % | 226 |
| S06 | R3 | 23,1 % | 49,5 % | 207 | 73,4 % | 159 |
| S06 | R0∪R2 | 21,4 % | 46,3 % | 206 | 69,5 % | 162 |
| S07 | R0 | 22,2 % | 38,2 % | 228 | 55,9 % | 192 |
| S07 | R1 | 30,0 % | 39,6 % | 44 | 51,7 % | 76 |
| S07 | R2 | 8,2 % | 13,2 % | 68 | 20,5 % | 175 |
| S07 | R3 | 24,8 % | 40,9 % | 225 | 58,1 % | 187 |
| S07 | R0∪R2 | 26,2 % | 42,1 % | 208 | 58,7 % | 180 |
| S08 | R0 | 20,5 % | 39,6 % | 224 | 56,8 % | 190 |
| S08 | R1 | 28,4 % | 40,4 % | 59 | 53,5 % | 93 |
| S08 | R2 | 6,8 % | 14,4 % | 142 | 21,9 % | 198 |
| S08 | R3 | 25,9 % | 46,2 % | 213 | 64,1 % | 175 |
| S08 | R0∪R2 | 23,2 % | 42,5 % | 211 | 58,8 % | 182 |
| S09 | R0 | 3,5 % | 17,4 % | 271 | 30,8 % | 245 |
| S09 | R1 | 30,7 % | 49,5 % | 56 | 64,5 % | 66 |
| S09 | R2 | 8,6 % | 22,2 % | 115 | 33,8 % | 177 |
| S09 | R3 | 25,3 % | 54,6 % | 198 | 74,9 % | 149 |
| S09 | R0∪R2 | 10,4 % | 27,6 % | 198 | 41,6 % | 196 |
| Book 8 | R0 | 0,0 % | 0,0 % | n/d | 0,0 % | n/d |
| Book 8 | R1 | 27,5 % | 72,8 % | 66 | 86,1 % | 52 |
| Book 8 | R2 | 6,6 % | 41,3 % | 172 | 61,4 % | 151 |
| Book 8 | R3 | 25,0 % | 87,0 % | 129 | 94,1 % | 120 |
| Book 8 | R0∪R2 | 6,6 % | 41,3 % | 172 | 61,4 % | 151 |

## Application glissante sur l'historique (R1 à R3)

R1 et R2 sont appliquées ici au maxDD de chaque fenêtre glissante de 250 séances, R3 au PnL glissant sur 120 séances. Une sleeve « restée positive ensuite » est une sleeve dont le PnL des 120 (ou 250) séances suivant le déclenchement est positif : c'est une fausse alarme ex post. Ces seuils sont calibrés sur le même historique, ce qui sous-estime les déclenchements.

| Série | Règle | Épisodes | Séances en alarme | Suivis de PnL > 0 à +120 | à +250 | Premiers déclenchements |
|---|---|---:|---:|---|---|---|
| S01 | R1 | 0 | 0 | 0/0 | 0/0 |  |
| S01 | R2 | 0 | 0 | 0/0 | 0/0 |  |
| S01 | R3 | 2 | 8 | 2/2 | 2/2 | 2021-02-16, 2023-09-27 |
| S01 | R0∪R2 | 0 | 0 | 0/0 | 0/0 |  |
| S02 | R1 | 0 | 0 | 0/0 | 0/0 |  |
| S02 | R2 | 0 | 0 | 0/0 | 0/0 |  |
| S02 | R3 | 3 | 110 | 3/3 | 2/2 | 2020-03-18, 2020-06-23, 2025-09-29 |
| S02 | R0∪R2 | 0 | 0 | 0/0 | 0/0 |  |
| S03 | R1 | 0 | 0 | 0/0 | 0/0 |  |
| S03 | R2 | 0 | 0 | 0/0 | 0/0 |  |
| S03 | R3 | 4 | 109 | 4/4 | 4/4 | 2020-11-11, 2020-12-30, 2024-04-16, 2024-04-23 |
| S03 | R0∪R2 | 0 | 0 | 0/0 | 0/0 |  |
| S04 | R1 | 0 | 0 | 0/0 | 0/0 |  |
| S04 | R2 | 0 | 0 | 0/0 | 0/0 |  |
| S04 | R3 | 5 | 11 | 3/3 | 3/3 | 2025-02-13, 2025-02-25, 2025-02-28, 2025-09-30, 2025-11-07 |
| S04 | R0∪R2 | 0 | 0 | 0/0 | 0/0 |  |
| S05 | R1 | 0 | 0 | 0/0 | 0/0 |  |
| S05 | R2 | 0 | 0 | 0/0 | 0/0 |  |
| S05 | R3 | 1 | 1 | 1/1 | 1/1 | 2020-09-08 |
| S05 | R0∪R2 | 1 | 1 | 1/1 | 1/1 | 2020-09-16 |
| S06 | R1 | 0 | 0 | 0/0 | 0/0 |  |
| S06 | R2 | 0 | 0 | 0/0 | 0/0 |  |
| S06 | R3 | 1 | 1 | 1/1 | 1/1 | 2024-11-22 |
| S06 | R0∪R2 | 1 | 4 | 1/1 | 0/0 | 2025-09-23 |
| S07 | R1 | 0 | 0 | 0/0 | 0/0 |  |
| S07 | R2 | 0 | 0 | 0/0 | 0/0 |  |
| S07 | R3 | 2 | 3 | 1/1 | 1/1 | 2025-05-14, 2026-08-18 |
| S07 | R0∪R2 | 0 | 0 | 0/0 | 0/0 |  |
| S08 | R1 | 0 | 0 | 0/0 | 0/0 |  |
| S08 | R2 | 0 | 0 | 0/0 | 0/0 |  |
| S08 | R3 | 4 | 7 | 0/0 | 0/0 | 2026-06-01, 2026-08-03, 2026-08-10, 2026-08-12 |
| S08 | R0∪R2 | 0 | 0 | 0/0 | 0/0 |  |
| S09 | R1 | 1 | 144 | 1/1 | 1/1 | 2025-08-13 |
| S09 | R2 | 0 | 0 | 0/0 | 0/0 |  |
| S09 | R3 | 10 | 76 | 4/4 | 3/3 | 2025-07-09, 2025-08-08, 2025-09-17, 2025-10-09, 2026-06-16, 2026-06-19 |
| S09 | R0∪R2 | 0 | 0 | 0/0 | 0/0 |  |
| Book 8 | R1 | 0 | 0 | 0/0 | 0/0 |  |
| Book 8 | R2 | 0 | 0 | 0/0 | 0/0 |  |
| Book 8 | R3 | 2 | 3 | 2/2 | 0/0 | 2025-08-18, 2025-09-25 |
| Book 8 | R0∪R2 | 0 | 0 | 0/0 | 0/0 |  |

## Réintégration proposée

Une sleeve coupée passe en shadow. Elle revient après N séances de shadow si son PnL cumulé en shadow n'est jamais passé sous le p5(h) et qu'il est ≥ 0 à N. Table : probabilité de réintégration.

| Série | N | P(réintégration), sleeve saine | SR vrai 0 | SR vrai −0,5 |
|---|---:|---:|---:|---:|
| S01 | 126 | 60,0 % | 27,5 % | 18,3 % |
| S01 | 252 | 58,7 % | 19,8 % | 10,2 % |
| S02 | 126 | 62,0 % | 38,6 % | 24,8 % |
| S02 | 252 | 63,6 % | 34,2 % | 18,6 % |
| S03 | 126 | 52,8 % | 39,8 % | 27,0 % |
| S03 | 252 | 55,1 % | 38,1 % | 21,0 % |
| S04 | 126 | 48,0 % | 34,1 % | 20,4 % |
| S04 | 252 | 48,4 % | 31,6 % | 15,2 % |
| S05 | 126 | 51,7 % | 35,9 % | 22,1 % |
| S05 | 252 | 52,5 % | 37,4 % | 19,8 % |
| S06 | 126 | 55,1 % | 30,9 % | 18,8 % |
| S06 | 252 | 56,8 % | 29,0 % | 14,7 % |
| S07 | 126 | 54,9 % | 41,1 % | 27,4 % |
| S07 | 252 | 56,0 % | 38,1 % | 21,1 % |
| S08 | 126 | 54,4 % | 36,8 % | 25,1 % |
| S08 | 252 | 56,9 % | 34,9 % | 21,9 % |
| S09 | 126 | 61,6 % | 38,6 % | 23,2 % |
| S09 | 252 | 62,9 % | 34,0 % | 15,7 % |
| Book 8 | 126 | 68,1 % | 27,7 % | 16,7 % |
| Book 8 | 252 | 64,6 % | 13,1 % | 5,4 % |

## Format de suivi hebdomadaire

Gabarit : [05_suivi_hebdo_gabarit.csv](05_suivi_hebdo_gabarit.csv). Une ligne par sleeve et par semaine ISO, plus une ligne « Book 8 ». Elle reprend : le PnL de la semaine, le cumul depuis le 2026-11-03, le roulant sur 126 séances et son écart au seuil R0, le maxDD depuis le début et le seuil R1/R2 à h, le roulant sur 120 séances et le seuil R3, l'alarme et un commentaire. Toute alarme déclenche une revue, jamais une action automatique.

## Lecture et choix qui vous revient

1. **Aucune règle ne détecte vite sans fausses alarmes nombreuses.** C'est la conséquence directe de la phase 2 : les sleeves ont des Sharpe de 0,4 à 1,3, donc 250 séances ne suffisent pas à distinguer une sleeve saine d'une sleeve à Sharpe nul.
2. **R0, la règle en vigueur, est lente sans être sûre.** Elle coupe à tort environ 17 % des sleeves saines en 250 séances (médiane), ne détecte un Sharpe vrai nul que 31 % du temps, et met environ 200 séances quand elle le fait. Son seuil « pire que tout l'historique » n'empêche pas les fausses alarmes, parce que l'historique contient moins de fenêtres indépendantes que le futur n'en tire.
3. **R2 (drawdown au-delà du p99) est la seule règle à faible taux de fausses alarmes** (environ 7 %). Elle rate en revanche la plupart des dégradations : 14 % de détection à SR 0, 23 % à SR −0,5.
4. **R1 et R3 détectent plus, mais coupent à tort une sleeve saine sur quatre environ.** R1 est la plus rapide (environ 80 séances), R3 la plus sensible (66 % de détection à SR −0,5).
5. **Au niveau du book, la détection est bien meilleure**, parce que son Sharpe est plus élevé : R3 détecte 87 % des books à Sharpe vrai nul en 250 séances, pour 25 % de fausses alarmes, et R2 en détecte 41 % pour 7 %.
6. **La réintégration par le shadow discrimine mal.** Après 252 séances, une sleeve saine n'est réintégrée que 48 à 65 % du temps, et une sleeve à Sharpe nul l'est encore 20 à 38 % du temps.
7. **Sur l'historique**, R1 aurait signalé **S09** dès le 2025-08-13, au début de son legacy OOS (144 séances en alarme), et R3 signale **S08** en juin-août 2026.

8. **Combiner R0 et R2 pour le passage en shadow coûte cher en fausses alarmes.** Avec la règle « R0 ou R2 » (ligne R0∪R2, ajoutée le 2026-11-15 sur les mêmes tirages), une sleeve saine est coupée à tort dans 21 % des cas (médiane), contre 7 % pour R2 seule. La détection monte à 36 % à SR 0 et à 59 % à SR −0,5. **Correction** : la première version de cette page annonçait environ 7 % pour un passage en shadow sur « R2 ou R0 ». Ce chiffre ne valait que pour R2 seule.

**Proposition, à décider par vous**, à deux niveaux, uniforme pour toutes les sleeves. Version corrigée le 2026-11-15 : R0 devient une alerte.

- **alerte « revue »** si R3 (PnL sur 120 séances < p5) ou R0 (règle déjà enregistrée) se déclenche, sans action automatique ;
- **passage en shadow** si R2 se déclenche (drawdown depuis le début du suivi au-delà du p99 à h) ;
- **réintégration** après 252 séances de shadow, par une décision versionnée, sur les critères de réintégration ci-dessus ;
- **au niveau du book**, R3 en alerte et R2 en revue générale.

Le compromis assumé : l'action automatique coupe peu de sleeves saines (environ 7 %), au prix d'une détection lente. Les alertes R3 et R0 servent à regarder plus tôt. Garder R0 comme déclencheur du shadow ferait passer les fausses coupures à 21 %.

Limites : les seuils sont estimés sur l'historique déjà vu, qui inclut la validation réutilisée. Le bootstrap suppose que la distribution passée reste valable. Pour S01, dont la fréquence de trades s'est effondrée en 2026 (phase 4), la distribution historique surestime la volatilité actuelle : une dégradation de S01 serait encore plus lente à détecter.
