# Référence opérationnelle — six sleeves verrouillées

> **Statut depuis le 4 octobre 2026 : baseline historique superseded.** La référence opérationnelle active est désormais [`ACTIVE_NINE_SLEEVES.md`](ACTIVE_NINE_SLEEVES.md). Les six sleeves ci-dessous restent gelées et sont conservées pour audit et comparaison.

Date de référence : **26 septembre 2026**.

## 1. Bloc MNQ

| Sleeve | Paramètres alpha verrouillés | Sortie / sizing | Statut |
|---|---|---|---|
| `orb_vwap_atr` | OR15 long-only, confirmation VWAP, filtre ATR(14), seuil de vote 0,50 | Stop sous l'opening range, cible 2R, sortie 16:00 ET; sizing nominal 0,5% sur 50k | LOCKED |
| `volatility_breakout` | vote de 42 cellules : ATR 10–22 pas 2 × multiple 0,50/0,60 × durée 270/300/360 | cible agrégée −1/0/+1 MNQ; seuil de vote 0,50 | LOCKED |
| `compression_breakout` | NR lookback 10, buffer 1 tick | 1 MNQ; sortie baseline verrouillée | LOCKED |
| `atr_trend_flip` | ATR 10, multiplicateur 1,75 | 1 MNQ; stop 1,0 ATR Wilder(14) 15 min | LOCKED |

Source canonique : [`locked_four_sleeve_config.json`](../research/locked_four_sleeve_portfolio/locked_four_sleeve_config.json).

### Allocation MNQ

La règle finale est `constrained_erc` : covariance rolling 63 séances connue à `t-1`, poids longs, somme 100%, sans levier, chaque sleeve bornée entre 10% et 40%.

Sur la fenêtre commune 2019-08-02–2026-03-19 :

| Allocation | PnL USD | Sharpe | PF journalier | Max DD USD |
|---|---:|---:|---:|---:|
| Equal weight | 13 134 | 1,195 | 1,270 | -1 386 |
| ERC rolling non contraint | 7 521 | 1,168 | 1,262 | -975 |
| **Constrained ERC retenu** | **10 532** | **1,299** | **1,279** | **-1 039** |

Ces chiffres sont historiques et ne constituent pas une promesse prospective.

## 2. Bloc MGC

| Sleeve | Paramètres verrouillés | IS / validation observés | Statut |
|---|---|---|---|
| `volatility_breakout` | `daily_atr_period=30`, `entry_atr_multiple=1.10` | Sharpe 0,431 / 0,485; PF 1,266 / 1,404; 134 / 65 trades | FREEZE CANDIDATE |
| `gap_continuation` | `minimum_gap_atr_multiple=0.50`, `volume_threshold_multiple=0.60` | Sharpe 0,872 / 0,864; PF 1,529 / 1,556; 141 / 95 trades | FREEZE CANDIDATE |

Source canonique : [`final_lock_config.json`](../research/mgc_final_two_sleeve_lock_20260926/final_lock_config.json).

Important : la période 2024-01-01–2026-03-19 a été réutilisée pendant les diagnostics MGC. Elle est donc adaptative et **n'est plus un OOS vierge**. La prochaine preuve utile doit être prospective, sans changement de paramètres.

## 3. Règles de maintenance

- Ne jamais sélectionner une variante après observation de nouvelles performances.
- Ne pas modifier les coûts, horaires, fills ou hard-flatten sans versionner une nouvelle spécification.
- Conserver les hashes des configurations et des données dans les rapports d'exécution.
- Suivre séparément le bloc MNQ et le bloc MGC, puis le portefeuille global, sans confondre allocation de capital et validation d'alpha.
- En cas de dégradation prospective, appliquer une règle de surveillance préspécifiée; ne pas réparer la stratégie par optimisation rétrospective.
