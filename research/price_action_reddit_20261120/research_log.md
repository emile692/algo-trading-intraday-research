# Journal de recherche : setups price action (post Reddit)

Chaque essai est noté ici, y compris les échecs, dans l'ordre où il a été fait.
Les règles sont dans [STRATEGIES.md](STRATEGIES.md). Le hors-échantillon
(2024-01-01 et après) n'a **jamais** été chargé : `run_is.py` coupe les données
au 2024-01-01 et vérifie que rien ne dépasse cette date.

## 2026-11-20

### 1. Données
- **Disque :** pas de ES, NQ et MES s'arrêtent au 2026-03-20, MNQ au 2026-10-02.
- **Problèmes des séries `c.0` :**
  - le contrat qui expire est gardé pendant la semaine du roll (volume à 25 % de la normale) ;
  - pas de barre RTH les vendredis d'échéance ;
  - faux gap de 200 à 270 points le jour du changement de contrat.
- **Décision de l'utilisateur :** étude sur micros uniquement. Téléchargement de `MNQ.v.0` et `MES.v.0` en 1 min, du 2019-05-01 au 2026-11-20 00:00 UTC, pour **19,44 USD**. Script : [download_micro_v0_full_20261120.py](../../scripts/download_micro_v0_full_20261120.py).

### 2. Code et tests, avant tout calcul sur données réelles
- Modules [src/price_action/](../../src/price_action/) (données, signaux, exécution, métriques) et 10 tests.
- **Deux fuites d'information trouvées dans ma première version, corrigées avant tout calcul :**
  - le setup A utilisait la validité du jour J entier (nombre de barres, contrat de la dernière barre), qu'on ne connaît qu'à la clôture ;
  - le setup B vérifiait l'unicité du contrat sur toute la journée.

  Les deux reposent désormais uniquement sur J-1, J-2 et les barres déjà closes.
- **Contrôle du test anti-fuite :** j'ai injecté une fuite volontaire (chaque bougie voit la clôture de la suivante). Le test la détecte bien.

### 3. Essai rapide sur MNQ, IS, paramètres par défaut (5 min, fin 11:00, N = 2)

| Setup | Trades IS (3 ans) | PF brut | PF net | Espérance nette |
|---|---|---|---|---|
| A | 36 | 0,74 | 0,69 | −0,08 R |
| B | 9 | 0,43 | 0,40 | −0,49 R |

**Constat :** avec ces règles, environ 12 trades par an (A) et 3 par an (B). Le critère « au moins 100 trades hors échantillon par instrument » (2024-01 → 2026-11, environ 2,9 ans) est **hors de portée par construction**, quel que soit le résultat.

**Décision, prise avant de voir la grille :** élargir la grille IS **descriptive** aux bougies de 1 et 3 min.
- L'unité de temps est l'un des paramètres libres prévus.
- A : bougie {1, 3, 5, 15} × fin {10:30, 11:00, 11:30}.
- B : bougie {1, 3, 5} × N {1, 2, 3} × fin {10:30, 11:00, 11:30}.
- Les valeurs par défaut restent figées ; tout autre choix revient à l'utilisateur, avant le passage hors échantillon.

### 4. Campagne IS complète (MNQ et MES)
`run_is.py` produit la grille (A : 12 cellules, B : 27 cellules, par instrument), la robustesse à ±20 %, les tableaux par année et par sens, et le Monte Carlo (10 000 ordres de trades). Le premier lancement a planté sur une cellule sans aucun trade : `summarize` est corrigé, rien d'autre ne change.

**Setup A**, paramètres par défaut :

| | Trades | PF net | Espérance nette |
|---|---|---|---|
| MNQ | 36 | 0,69 | −0,08 R |
| MES | 30 | 1,71 | +0,10 R |

- Aucune cellule ne dépasse 46 trades en 3 ans.
- MNQ et MES se contredisent : à partir de 3 min, MNQ est sous 1 (sauf une cellule à 1,03) et MES au-dessus.
- 2023 est négative sur les deux instruments.
- **Conclusion : ne peut pas atteindre 100 trades hors échantillon.**

**Setup B**, paramètres par défaut : 9 et 8 trades. Sur les 17 cellules à au moins 100 trades par instrument :
- MNQ : PF net de 0,73 à 1,02 ;
- MES : PF net de 0,61 à 0,89, avec un coût de 0,24 R par trade en 1 min.
- **Conclusion : pas d'edge IS.**

**Décision :** aucune cellule n'est retenue, et les valeurs par défaut restent figées. Je **recommande de clore les deux setups sans lire le hors-échantillon**. Le passage hors échantillon n'est fait que si l'utilisateur valide STRATEGIES.md et le demande. Bilan détaillé : [REPORT.md](REPORT.md).

## 2026-11-21

### 5. Passage hors échantillon unique, demandé par l'utilisateur
- STRATEGIES.md est figé tel quel, avec les valeurs par défaut et les empreintes SHA-256 du code prises avant le passage.
- `run_oos.py` traite le 2024-01-03 → 2026-11-19 et refuse de tourner une seconde fois.

**Résultat :**

| | Trades | PF net | Espérance nette |
|---|---|---|---|
| A · MNQ | 49 | 1,53 | +0,25 R |
| A · MES | 42 | 0,77 | −0,01 R |
| B · MNQ | 12 | 1,01 | −0,30 R |
| B · MES | 16 | 1,24 | −0,06 R |

**Verdict : les deux setups sont rejetés.** Aucun instrument n'atteint 100 trades. Sur A, MNQ et MES ont échangé leurs rôles par rapport à l'IS, ce qui confirme que les deux résultats étaient du bruit. Rien n'a été modifié après la lecture.
