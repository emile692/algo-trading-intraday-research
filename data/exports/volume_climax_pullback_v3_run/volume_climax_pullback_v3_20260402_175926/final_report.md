# Volume Climax Pullback V3 - Final Report

## Scope
- Universe: `MNQ`, `MES`, `M2K`, `MGC`.
- Timeframe: `1h` RTH only, one open position max, flat end of session, leak-free convention inherited from V2.
- Tested V3 rows: `258`.
- V2 reference run: `data\exports\volume_climax_pullback_v2_run\volume_climax_pullback_v2_20260401_214553`.

## Verdicts By Asset
### MNQ
- Verdict: `recommandee`.
- Reason: Survivant propre OOS avec profil encore comparable a la reference V2.
- Recommended spec: `dynamic_exit_atr_target_1p0_ts2_vq0p95_bf0p5_ra1p2 | exit=atr_fraction ts=2 | vq=0.950 bf=0.5 ra=1.2 | regime=off/off/off`.
- OOS Sharpe / PF / Net PnL / Trades: `1.458` / `1.753` / `3038.57` / `101`.
- Stability IS/OOS: `1.165`.
- Vs V2 `dynamic_exit_atr_target_1p0_ts3_vq0p95_bf0p5_ra1p2`: delta Sharpe `0.160`, delta Net PnL `294.57`.
- Clean survivors / live variants: `48` / `48`.
### MES
- Verdict: `recommandee`.
- Reason: Survivant propre OOS avec profil encore comparable a la reference V2.
- Recommended spec: `dynamic_exit_mixed_ts4_vq0p95_bf0p6_ra1p5 | exit=mixed ts=4 | vq=0.950 bf=0.6 ra=1.5 | regime=off/off/off`.
- OOS Sharpe / PF / Net PnL / Trades: `1.406` / `2.213` / `1388.59` / `65`.
- Stability IS/OOS: `6.188`.
- Vs V2 `dynamic_exit_mixed_ts4_vq0p95_bf0p6_ra1p2`: delta Sharpe `0.164`, delta Net PnL `104.88`.
- Clean survivors / live variants: `48` / `48`.
### M2K
- Verdict: `recommandee`.
- Reason: Survivant propre OOS avec profil encore comparable a la reference V2.
- Recommended spec: `dynamic_exit_atr_target_1p0_ts4_vq0p95_bf0p5_ra1p2 | exit=atr_fraction ts=4 | vq=0.950 bf=0.5 ra=1.2 | regime=off/off/off`.
- OOS Sharpe / PF / Net PnL / Trades: `1.567` / `1.801` / `986.65` / `99`.
- Stability IS/OOS: `4.610`.
- Vs V2 `dynamic_exit_atr_target_1p0_ts3_vq0p95_bf0p5_ra1p2`: delta Sharpe `0.052`, delta Net PnL `30.00`.
- Clean survivors / live variants: `48` / `48`.
### MGC
- Verdict: `recommandee`.
- Reason: Survivant propre OOS avec profil encore comparable a la reference V2.
- Recommended spec: `regime_filtered_ema_mild_atr_20_80_compression_off_atr_target_1p0_ts3_vq0p95_bf0p5_ra1p2 | exit=atr_fraction ts=3 | vq=0.950 bf=0.5 ra=1.2 | regime=mild/20_80/off`.
- OOS Sharpe / PF / Net PnL / Trades: `1.184` / `3.948` / `1515.20` / `34`.
- Stability IS/OOS: `5.747`.
- Vs V2 `regime_filtered_trend_ema50_medium_vq0p975_bf0p5_ra1p2`: delta Sharpe `0.392`, delta Net PnL `587.70`.
- Clean survivors / live variants: `97` / `107`.

## Research Answers
1. Dynamic exit stabilization without losing the core PnL: `yes`.
2. Best exit_mode / time_stop by asset: `MNQ=atr_fraction / ts2`, `MES=mixed / ts4`, `M2K=atr_fraction / ts4`, `MGC=atr_fraction / ts3`.
3. MGC regime filter value: `positive`. Le filtre de regime ajoute une vraie valeur nette sur MGC face au meilleur dynamic_exit pur.
4. V2 survivors under the stricter grid: `12/12` explicit dynamic anchors remain alive. The historical MGC fixed-RR regime winner is kept as comparison reference only.
5. Recommended specs with few concurrent contenders: `no`.

## Anomalies / Faux Survivants
- 0 trade OOS: `0`.
- Profit factor inf: `0`.
- Echantillons trop petits (<6 trades OOS): `9`.
- Ratios de stabilite non definis: `1`.
- Examples small OOS samples: `MGC:regime_filtered_ema_mild_atr_20_80_compression_mild_atr_target_1p0_ts3_vq0p975_bf0p5_ra1p2 (4 trades)`, `MGC:regime_filtered_ema_mild_atr_30_70_compression_mild_atr_target_1p0_ts3_vq0p95_bf0p5_ra1p2 (3 trades)`, `MGC:regime_filtered_ema_mild_atr_30_70_compression_mild_mixed_ts4_vq0p95_bf0p5_ra1p2 (3 trades)`, `MGC:regime_filtered_ema_mild_atr_30_70_compression_mild_mixed_ts3_vq0p95_bf0p5_ra1p2 (3 trades)`, `MGC:regime_filtered_ema_mild_atr_20_80_compression_mild_mixed_ts3_vq0p975_bf0p5_ra1p2 (4 trades)`.
- Examples undefined stability: `MGC:regime_filtered_ema_mild_atr_off_compression_mild_mixed_ts4_vq0p975_bf0p5_ra1p2`.

## Global Conclusion
- Conclusion: `strategie prete pour paper trading cible`.
- Paper-trading readiness rule: at least three assets recommended, zero asset fully rejected, and at least four clean survivors overall.