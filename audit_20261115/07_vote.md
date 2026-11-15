# Phase 7 — Diagnostic du vote seuillé (EXPLORATOIRE)

*Aucune de ces variantes ne remplace la référence. Les conclusions s'appuient sur l'IS ; la validation et le total sont seulement reportés. Script : [phase7_vote.py](phase7_vote.py). Tables : [07_vote.csv](outputs/07_vote.csv), [07_s09_bridge.json](outputs/07_s09_bridge.json). PnL net en USD, coûts nominaux, Sharpe √252 sur le calendrier de la sleeve.*

## Gate de reproduction

Le vote est ré-implémenté de façon vectorisée, avec une fonction cible injectable. Avec la cible de référence, il doit redonner le PnL quotidien gelé de chaque sleeve avant toute variante.

| Sleeve | Séances | Écart max vs PnL gelé | Gate |
|---|---:|---:|---|
| S02 | 1747 | 0.00e+00 USD | réussi |
| S03 | 1747 | 0.00e+00 USD | réussi |
| S04 | 1747 | 0.00e+00 USD | réussi |
| S05 | 1775 | 1.46e-11 USD | réussi |
| S06 | 1775 | 1.27e-11 USD | réussi |
| S07 | 1085 | 2.22e-16 USD | réussi |
| S08 | 1085 | 2.84e-14 USD | réussi |
| S09 | 1085 | 7.11e-15 USD | réussi |

## Variantes

- **(a)** Vote seuillé de la référence : +1 si score ≥ 0,50, −1 si score ≤ −0,50, 0 sinon. Exécution à l'open de la minute suivante.
- **(b)** Moyenne des PnL quotidiens nets des cellules, chacune avec son propre stop et ses propres coûts. C'est une référence théorique : la répliquer demanderait des fractions de contrat.
- **(c)** q = arrondi(K × score) contrats, l'arrondi s'éloignant de zéro, pour K = 2 et 3, sur MNQ et MGC seulement. Coûts comptés par contrat à chaque changement de quantité.

| Sleeve | Variante | PnL IS | Sharpe IS | PnL validation | Sharpe validation | PnL total | Sharpe total | maxDD IS | Contrats IS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S02 | (a) vote seuillé | 10 980 | 0,859 | 9 279 | 0,899 | 20 259 | 0,848 | −4 790 | 1 452 |
| S02 | (b) moyenne des cellules | 8 893 | 0,786 | 9 234 | 0,989 | 18 127 | 0,846 | −4 061 | — |
| S02 | (c) K=2 | 18 364 | 0,800 | 20 147 | 1,063 | 38 510 | 0,885 | −8 750 | 2 706 |
| S02 | (c) K=3 | 28 080 | 0,798 | 28 148 | 0,974 | 56 228 | 0,846 | −13 244 | 4 162 |
| S03 | (a) vote seuillé | 1 663 | 0,546 | 592 | 0,283 | 2 256 | 0,432 | −1 037 | 438 |
| S03 | (b) moyenne des cellules | 1 500 | 0,527 | 739 | 0,361 | 2 239 | 0,448 | −851 | — |
| S03 | (c) K=2 | 1 256 | 0,222 | 1 412 | 0,311 | 2 668 | 0,253 | −2 152 | 872 |
| S03 | (c) K=3 | 3 174 | 0,372 | 2 046 | 0,311 | 5 220 | 0,336 | −3 029 | 1 310 |
| S04 | (a) vote seuillé | 4 692 | 0,354 | 5 383 | 0,543 | 10 075 | 0,424 | −3 063 | 2 838 |
| S04 | (b) moyenne des cellules | 5 096 | 0,406 | 5 006 | 0,578 | 10 102 | 0,468 | −3 126 | — |
| S04 | (c) K=2 | 5 306 | 0,204 | 4 374 | 0,267 | 9 680 | 0,226 | −7 971 | 5 846 |
| S04 | (c) K=3 | 10 472 | 0,273 | 10 058 | 0,394 | 20 530 | 0,317 | −10 666 | 8 714 |
| S05 | (a) vote seuillé | 1 418 | 0,519 | 1 718 | 0,388 | 3 136 | 0,370 | −1 346 | 264 |
| S05 | (b) moyenne des cellules | 1 069 | 0,442 | 324 | 0,112 | 1 393 | 0,236 | −994 | — |
| S05 | (c) K=2 | 2 636 | 0,563 | 970 | 0,148 | 3 606 | 0,280 | −1 710 | 598 |
| S05 | (c) K=3 | 4 163 | 0,582 | 924 | 0,098 | 5 087 | 0,272 | −2 785 | 880 |
| S06 | (a) vote seuillé | 3 098 | 0,878 | 3 932 | 0,732 | 7 030 | 0,677 | −962 | 276 |
| S06 | (b) moyenne des cellules | 2 969 | 0,861 | 3 517 | 0,664 | 6 487 | 0,635 | −867 | — |
| S06 | (c) K=2 | 5 838 | 0,866 | 5 656 | 0,529 | 11 494 | 0,560 | −1 759 | 562 |
| S06 | (c) K=3 | 8 710 | 0,853 | 9 571 | 0,599 | 18 281 | 0,594 | −2 597 | 854 |
| S07 | (a) vote seuillé | 1 135 | 0,390 | 947 | 0,638 | 2 082 | 0,473 | −1 286 | 636 |
| S07 | (b) moyenne des cellules | 758 | 0,262 | 1 112 | 0,760 | 1 871 | 0,428 | −1 373 | — |
| S08 | (a) vote seuillé | 2 535 | 0,621 | 1 365 | 0,687 | 3 900 | 0,642 | −1 543 | 1 560 |
| S08 | (b) moyenne des cellules | 1 909 | 0,490 | 1 429 | 0,742 | 3 338 | 0,573 | −1 512 | — |
| S09 | (a) vote seuillé | 4 989 | 1,314 | −73 | −0,040 | 4 916 | 0,874 | −985 | 894 |
| S09 | (b) moyenne des cellules | 4 376 | 1,321 | 920 | 0,559 | 5 296 | 1,066 | −794 | — |

## S09 — de (a) à (b) sur le legacy OOS (2025-05-29 → 2026-10-02)

Décomposition additive. Les étapes 3a à 3d comparent, minute par minute, une exposition linéaire (q = score, fractionnaire) au vote, tous deux sans coût ni décalage.

| Étape | USD |
|---|---:|
| (a) vote seuillé, net | −73,22 |
| + 1. coûts du vote | 844,22 |
| + 2. décalage d'une minute (open suivant → close de la barre du signal) | 87,50 |
| + 3a. jours où le vote ne se déclenche jamais | −1 465,63 |
| + 3b. minutes avant l'entrée du vote | 948,79 |
| + 3c. minutes après une sortie anticipée (score repassé sous ±0,50) | −16,71 |
| + 3d. exposition linéaire − exposition du vote, de la première entrée à la dernière sortie | 1 038,42 |
| = exposition linéaire (q = score), sans coût ni décalage | 1 363,38 |
| + 4. résidu : exécution propre des cellules (stops intrabarre) et leurs coûts | −443,62 |
| = (b) moyenne des PnL des cellules | 919,76 |

Jours où plus de la moitié des cellules gagnent alors que le vote reste flat toute la séance : **0** (sur 107 jours de vote flat), pour 0,00 USD de PnL moyen des cellules.

## Lecture

Conclusions fondées sur l'IS uniquement.

1. **Le vote seuillé ne détruit pas de valeur sur l'IS.** Il fait mieux que la moyenne de ses cellules (b) pour 7 sleeves sur 8, en PnL (S02 10 980 contre 8 893 USD, S08 2 535 contre 1 909, S09 4 989 contre 4 376) et le plus souvent en Sharpe. Seule S04 fait moins bien que ses cellules (4 692 contre 5 096 USD). L'idée d'un vote qui « laisse de l'argent sur la table » n'est pas confirmée sur la fenêtre de sélection.
2. **La taille proportionnelle au score (c) n'améliore pas le rendement ajusté du risque sur l'IS.** Le PnL augmente avec K, mais le drawdown et les coûts augmentent davantage :
   - S02 : Sharpe 0,86 → 0,80, avec un maxDD IS qui passe de −4 790 à −8 750 USD à K = 2 ;
   - S03 et S04 : le Sharpe est divisé par 1,3 à 2,5 ;
   - S06 est stable ;
   - seule S05 progresse un peu (0,52 → 0,56–0,58), puis s'effondre sur la validation (0,39 → 0,10–0,15).

   **Aucune variante (c) n'est retenue.** Pour un compte Topstep, elle multiplierait aussi les lots MNQ/MGC consommés.
3. **S09 sur son legacy OOS.** L'écart de 993 USD entre (b) et (a) se décompose ainsi :
   - les **coûts du vote** : 844 USD. Le vote change souvent de cible pour un gain par trade faible ;
   - le **décalage d'une minute** : 88 USD ;
   - le **seuil**, net : +505 USD. Le vote entre tard (+949 USD captés par l'exposition linéaire avant son entrée) et prend une exposition pleine (+1 038 USD de mieux pour l'exposition fractionnaire pendant le vote). Mais il évite les jours faibles : ne jamais se déclencher fait économiser 1 466 USD. Les sorties anticipées ne coûtent presque rien (−17 USD) ;
   - l'exécution propre des cellules (stops intrabarre) et leurs coûts : −444 USD.

   Aucun jour n'a vu la majorité des cellules gagner pendant que le vote restait flat. Le vote nul de S09 sur son OOS s'explique d'abord par ses **coûts de rotation**, puis par le retard d'entrée. Ce n'est pas un défaut d'agrégation qu'un autre seuil corrigerait proprement.

Toute modification du vote serait une nouvelle hypothèse, à pré-enregistrer et à juger sur des données neuves.
