# Noise Gate Validation Summary

## Setup

- Campagne ciblee: baseline figee + overlay noise-area gate uniquement.
- Pas de compression filter, pas de VWAP exit, pas de full re-opt global.
- Classement principal: `ranking_prop_score` (prop-firm oriented, explicite).

## Baseline vs Best Overlay

- Baseline (`baseline_fixed`): prop_score=-0.7511, net_pnl=6343.00, maxDD=-1177.50.
- Best noise gate (`noise_gate__max_or_noise__L20__vm1p0__k0p0__c1__continuous_on_bar_close`): prop_score=1.7650, net_pnl=4433.50, maxDD=-621.00.
- Delta best-baseline: d_prop=2.5161, d_net_pnl=-1909.50, d_maxDD=556.50.

## Heatmap Read

- Cluster LxVM (style `max_or_noise`): fragile (best=0.927, voisins>=90%=2).
- Sensibilite confirm_bars x schedule autour du meilleur couple structurel: spread confirm/schedule=1.529.

## Top Robust Parameters

- Recommande (validation / production research): `noise_gate__max_or_noise__L20__vm1p0__k0p0__c1__continuous_on_bar_close`.
- Preset safe / prop-firm oriented: `noise_gate__max_or_noise__L20__vm1p25__k0p0__c1__continuous_on_bar_close`.

## Decision (Required Questions)

1. Est-ce que le noise-area gate ameliore la baseline figee ? Oui.
2. Quel reglage est le meilleur selon prop_score ? `noise_gate__max_or_noise__L20__vm1p0__k0p0__c1__continuous_on_bar_close`.
3. Quel reglage est le plus robuste visuellement via heatmaps ? `noise_gate__max_or_noise__L20__vm1p0__k0p0__c1__continuous_on_bar_close` (attention: cluster peu dense).
4. Y a-t-il un cluster simple et stable autour du meilleur reglage ? Non / a confirmer.
5. Recommandes-tu d'ajouter cette option a la baseline ? Oui, en option via flag explicite.
6. Quel preset recommandes-tu comme preset validation / production research ? `noise_gate__max_or_noise__L20__vm1p0__k0p0__c1__continuous_on_bar_close` (safe alternatif: `noise_gate__max_or_noise__L20__vm1p25__k0p0__c1__continuous_on_bar_close`).

## Notes

- La baseline reste intacte: verification de non-regression executee avant conclusion.
- Les resultats complets sont dans `noise_gate_validation_results.csv` et `noise_gate_top_configs.csv`.
