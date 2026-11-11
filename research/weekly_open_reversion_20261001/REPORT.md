# Tuesday → Weekly Open — MNQ/NQ

## Verdict

L'observation « retour vers le Weekly Open » existe nettement, mais **le taux de touch seul surestime fortement la qualité tradable du mouvement**.

- À moins de 0,25 % du Weekly Open, le target est touché dans 87–100 % des cas selon instrument et direction. Pourtant, il n'est touché avant une excursion adverse de 1D que dans 31–47 % des cas.
- Entre 0,25 % et 1,00 %, le taux de touch se situe généralement entre 55 % et 81 %, avec 46–60 % de touches avant 1D adverse.
- Au-delà de 1,00 %, le taux de touch tombe autour de 27–36 %. Cette zone n'est pas soutenue comme setup de reversion systématique à horizon mercredi.
- Le PnL moyen sans stop n'est pas uniformément positif, même dans les buckets où le taux de touch est élevé : quelques trajectoires non revenues dominent les nombreuses petites touches.

Il s'agit d'un diagnostic descriptif, sans coûts, slippage ni modèle d'exécution. MNQ et NQ suivent le même sous-jacent et ne doivent pas être additionnés comme deux échantillons indépendants.

## Résultats NQ

Historique : 7 juin 2010 – 16 mars 2026, 783 semaines incluses.

| Direction | Distance | n | Touch target (IC 95 %) | Avant 1D | Avant 2D | MAE médiane | Délai médian si touch | PnL moyen mar. 11:30 / 16:00 / mer. 16:00 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Long | 0–0,25 % | 68 | 98,5 % (92,1–99,7) | 47,1 % | 73,5 % | 1,05D | 0,05 h | +0,30D / +0,55D / +0,93D |
| Long | 0,25–0,50 % | 64 | 71,9 % (59,9–81,4) | 53,1 % | 64,1 % | 0,88D | 0,86 h | −0,04D / −0,09D / −0,04D |
| Long | 0,50–1,00 % | 80 | 70,0 % (59,2–78,9) | 52,5 % | 66,2 % | 0,70D | 3,43 h | +0,08D / +0,14D / +0,30D |
| Long | >1,00 % | 104 | 33,7 % (25,3–43,2) | 30,8 % | 33,7 % | 0,55D | 23,00 h | +0,06D / +0,04D / +0,17D |
| Short | 0–0,25 % | 105 | 91,4 % (84,5–95,4) | 40,0 % | 61,9 % | 1,47D | 0,05 h | −0,04D / −0,13D / −0,09D |
| Short | 0,25–0,50 % | 83 | 57,8 % (47,1–67,9) | 45,8 % | 55,4 % | 1,07D | 0,71 h | −0,19D / −0,25D / −0,45D |
| Short | 0,50–1,00 % | 145 | 55,2 % (47,0–63,0) | 48,3 % | 54,5 % | 0,71D | 4,16 h | +0,08D / +0,04D / +0,09D |
| Short | >1,00 % | 134 | 30,6 % (23,4–38,8) | 29,1 % | 30,6 % | 0,58D | 9,18 h | −0,03D / +0,04D / +0,07D |

## Résultats MNQ

Historique : 6 mai 2019 – 16 mars 2026, 344 semaines incluses.

| Direction | Distance | n | Touch target (IC 95 %) | Avant 1D | Avant 2D | MAE médiane | Délai médian si touch | PnL moyen mar. 11:30 / 16:00 / mer. 16:00 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Long | 0–0,25 % | 35 | 100,0 % (90,1–100,0) | 42,9 % | 68,6 % | 1,07D | 0,03 h | +0,06D / +0,47D / +1,00D |
| Long | 0,25–0,50 % | 27 | 70,4 % (51,5–84,1) | 55,6 % | 55,6 % | 0,88D | 0,42 h | +0,08D / +0,03D / −0,07D |
| Long | 0,50–1,00 % | 32 | 81,2 % (64,7–91,1) | 53,1 % | 71,9 % | 0,70D | 2,44 h | +0,09D / +0,22D / +0,60D |
| Long | >1,00 % | 45 | 26,7 % (16,0–41,0) | 22,2 % | 26,7 % | 0,74D | 24,22 h | −0,18D / −0,17D / +0,02D |
| Short | 0–0,25 % | 39 | 87,2 % (73,3–94,4) | 30,8 % | 56,4 % | 1,61D | 0,05 h | −0,41D / −0,31D / −1,01D |
| Short | 0,25–0,50 % | 27 | 66,7 % (47,8–81,4) | 59,3 % | 66,7 % | 0,45D | 0,27 h | +0,18D / +0,06D / −0,25D |
| Short | 0,50–1,00 % | 62 | 66,1 % (53,7–76,7) | 59,7 % | 66,1 % | 0,56D | 4,00 h | +0,18D / +0,19D / +0,22D |
| Short | >1,00 % | 77 | 36,4 % (26,5–47,5) | 33,8 % | 36,4 % | 0,58D | 12,83 h | −0,00D / +0,08D / +0,15D |

## Cohérence MNQ/NQ

Sur les 344 semaines communes, le résultat `target_hit` est identique dans 100 % des cas. La direction est identique dans 343/344 cas, le classement de distance dans 344/344, et les résultats « avant 1D » et « avant 2D » concordent respectivement dans 342/344 et 341/344 cas. Cette réplication confirme surtout la cohérence des données; elle ne constitue pas une seconde validation indépendante.

## Protocole figé

- Weekly Open : `open` de la barre minute du lundi 09:30 ET.
- Entrée : `open` de la barre minute du mardi 09:30 ET, direction vers le Weekly Open.
- `D` : distance absolue entre entrée et target.
- Horizon : mercredi 16:00 ET; Globex entre mardi et mercredi est inclus.
- Checkpoints PnL : close de la barre 11:29 et 15:59 du mardi, puis 15:59 du mercredi. Le PnL est figé à +1D si le target a déjà été touché; aucun stop adverse n'est appliqué.
- MAE : excursion adverse jusqu'à la barre du target incluse, ou jusqu'à l'horizon si non touché. Cette convention est conservatrice sur la barre de target.
- MFE : le ledger contient à la fois la MFE avant target et la MFE complète jusqu'à mercredi.
- Si target et barrière adverse sont touchés dans la même minute, le target n'est pas compté « avant » la barrière.
- Une semaine est exclue si un timestamp obligatoire manque, si D est nul/invalide ou si le contrat continu change entre les anchors et la fin de fenêtre.
- Buckets : `[0;0,25[`, `[0,25;0,50[`, `[0,50;1,00]`, `]1,00;+∞[`.

Les intervalles à 95 % sont des intervalles de Wilson. Ils décrivent l'incertitude d'échantillonnage simple; ils ne corrigent ni l'autocorrélation temporelle ni les changements de régime.

## Fichiers

- `weekly_observations.csv` : ledger semaine par semaine, toutes les mesures en points et en unités de D.
- `summary_by_symbol_direction_bucket.csv` : agrégats et intervalles de confiance.
- `exclusion_audit.csv` : chaque semaine incluse ou exclue et la raison.
- `run_metadata.json` : hypothèse, conventions et chemins d'entrée.
