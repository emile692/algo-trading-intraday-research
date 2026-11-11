Spec git commit: `aa5bd8ef005a282f60517865e8ce0f347f429f49`

# Phase 2 — pré-enregistrement et tests gelés

## Audit de pré-enregistrement

Tous les calculs ci-dessous utilisent [phase2_spec.json](phase2_spec.json) tel que committé dans `aa5bd8ef005a282f60517865e8ce0f347f429f49`. Le runner vérifie le blob git, l’absence de diff et les SHA-256 des sources avant exécution. Aucun full sample n’est utilisé comme critère.

Le premier commit de gel contenait deux erreurs de transcription de SHA-256 (NQ et M2K). Le garde-fou a interrompu le runner avant toute lecture analytique OOS. Seules ces deux empreintes ont été corrigées, sans changement de paramètre ni de protocole, puis la spec a été regelée au commit affiché en tête.

Divulgation : avant le commit, deux lignes individuelles 2024 du ledger déjà gelé de la sleeve MGC volatility breakout ont été affichées pendant l’identification de la source. Aucun agrégat ni résultat A/B n’avait été calculé. Les seuils génériques ont ensuite été gelés avant les tests.

## A — NQ principal

Le stop A0 existant était `max(distance entrée→target, ATR intraday causal du signal)`. A-stop le remplace par exactement la distance entrée→target ; toutes les autres règles restent identiques. La réplication A0 est exacte trade par trade avant comparaison.

| variant | period | trades | valid_R_trades | mean_net_R | trade_CVaR5_loss_R | max_drawdown_loss_R | net_R | net_usd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | IS | 1954 | 1946 | -0.0211 | 0.6939 | 44.3237 | -40.9987 | -83403.2143 |
| A0 | OOS | 304 | 304 | 0.0361 | 0.5979 | 4.3018 | 10.9681 | 77970.0000 |
| A_stop | IS | 1954 | 1946 | -0.0211 | 0.6939 | 44.3851 | -40.9771 | -80860.0000 |
| A_stop | OOS | 304 | 304 | 0.0330 | 0.5979 | 4.5651 | 10.0398 | 71555.0000 |

Décision pré-enregistrée : CVaR IS améliorée = `False` ; max drawdown IS amélioré = `False` ; non-dégradation OOS = `True`. Variante forward retenue : **A0**.

Concrètement, la CVaR 5 % IS est inchangée et le max drawdown IS est légèrement plus mauvais avec A-stop : la variante échoue donc le gate IS, indépendamment de son passage du gate OOS.

## A — micros en robustesse seulement

| asset | variant | period | trades | mean_net_R | trade_CVaR5_loss_R | max_drawdown_loss_R | net_R | net_usd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MNQ | A0 | IS | 711 | -0.0142 | 0.6521 | 14.3328 | -9.9536 | -6870.2500 |
| MNQ | A0 | OOS | 304 | 0.0311 | 0.6017 | 4.5923 | 9.4503 | 6844.0357 |
| MNQ | A_stop | IS | 711 | -0.0133 | 0.6521 | 13.7635 | -9.3052 | -6554.0000 |
| MNQ | A_stop | OOS | 304 | 0.0296 | 0.6017 | 4.8603 | 9.0025 | 6500.5000 |
| MES | A0 | IS | 746 | -0.0173 | 0.6271 | 16.6744 | -12.6881 | -3180.6250 |
| MES | A0 | OOS | 297 | 0.0104 | 0.6170 | 6.5784 | 3.1004 | 2178.3036 |
| MES | A_stop | IS | 746 | -0.0170 | 0.6271 | 16.5168 | -12.4962 | -3153.7500 |
| MES | A_stop | OOS | 297 | 0.0082 | 0.6170 | 6.9187 | 2.4216 | 1981.2500 |
| M2K | A0 | IS | 678 | 0.0000 | 0.6411 | 12.6584 | 0.0270 | -673.0714 |
| M2K | A0 | OOS | 293 | 0.0191 | 0.5155 | 3.9972 | 5.6011 | 1542.2857 |
| M2K | A_stop | IS | 678 | 0.0006 | 0.6411 | 13.0320 | 0.4133 | -622.0000 |
| M2K | A_stop | OOS | 293 | 0.0179 | 0.5155 | 4.0767 | 5.2587 | 1482.0000 |

Les micros ne participent jamais à la sélection.

## B — deux séries de suivi

| version | period | trades | valid_R_trades | mean_net_R | sd_net_R | net_R | net_usd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B_canonical | IS | 1333 | 1333 | -0.0357 | 0.1790 | -47.5227 | -6762.5000 |
| B_canonical | OOS | 318 | 318 | 0.0085 | 0.1661 | 2.6937 | 3165.0000 |
| B_no_volume | IS | 1896 | 1896 | -0.0324 | 0.1718 | -61.3788 | -8721.0000 |
| B_no_volume | OOS | 547 | 547 | 0.0018 | 0.1703 | 0.9700 | 2599.5000 |

`B_no_volume` retire uniquement le seuil RVOL ; le mouvement, la confirmation, l’entrée, les coûts, le holding et le warm-up restent gelés. Aucune variante B n’est sélectionnée.

## Protocole forward calibré sur l’OOS

| series | OOS_trades | OOS_mean_net_R | OOS_sd_net_R | review_N | abandon_threshold_mean_net_R | finite_power_review | forward_start | kill_switch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_retained | 304 | 0.0361 | 0.3225 | 361 | 0.0143 | True | 2026-03-20 | trailing_365_calendar_day_net_R_strictly_negative_after_365_elapsed_days |
| B_canonical | 318 | 0.0085 | 0.1661 | 1734 | 0.0034 | True | 2026-09-23 | trailing_365_calendar_day_net_R_strictly_negative_after_365_elapsed_days |
| B_no_volume | 547 | 0.0018 | 0.1703 | 41587 | 0.0007 | True | 2026-09-23 | trailing_365_calendar_day_net_R_strictly_negative_after_365_elapsed_days |

La revue est unique au premier passage de N trades. Le seuil d’abandon est la moyenne forward indiquée dans le tableau. Le kill-switch commun est un PnL net R strictement négatif sur les 365 jours calendaires précédents, évalué après chaque trade seulement une fois 365 jours écoulés depuis le début forward.

## B contre la sleeve MGC volatility breakout — OOS uniquement

| common_OOS_start | common_OOS_end | common_available_sessions | B_trades | volatility_trades | pearson_daily_net_usd | spearman_daily_net_usd | pearson_union_active_days | same_session_count | B_active_sessions | volatility_active_sessions | same_session_rate_over_B | same_session_rate_over_volatility | B_position_overlap_rate | volatility_position_overlap_rate | decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-03-17 | 2026-09-22 | 908 | 318 | 93 | 0.2799 | -0.0962 | 0.2777 | 48 | 282 | 93 | 0.1702 | 0.5161 | 0.1226 | 0.3978 | inconclusive_complementary_diagnostic |

La décision applique mécaniquement les seuils de la spec : **inconclusive_complementary_diagnostic**. Aucune performance conditionnée ou filtrée n’est calculée.

Justification exacte : le statut `separate_sleeve` échoue parce que 51,6 % des sessions actives de volatility breakout coïncident avec une session B (seuil < 25 %), tandis que le statut `filter_only_candidate` échoue car la corrélation de Pearson (0,280), le recouvrement des sessions B (17,0 %) et le recouvrement des positions B (12,3 %) restent tous sous leurs seuils de 50 %.

## Conclusion

- Diagnostic clos conservé : A = (c2), B = (c1), hypothèse de mesure (b) rejetée.
- Phase 2 A : **A0** est la seule série forward retenue selon les deux gates IS et le gate OOS.
- Phase 2 B : les versions canonique et sans volume restent deux séries de mesure, sans sélection de règle.
- Rôle de B par rapport à MGC volatility breakout : **inconclusive_complementary_diagnostic** selon les seuils pré-enregistrés.
