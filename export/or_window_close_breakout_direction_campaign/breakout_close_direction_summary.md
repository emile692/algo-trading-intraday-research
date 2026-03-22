# OR Window Breakout Close Direction Campaign

Regle d'eligibilite retenue: journee conservee uniquement si l'Opening Range est complete pour la fenetre testee et si la session possede un close de reference coherent a `16:00` ou, a defaut, un dernier bar RTH entre `15:55` et `16:00`.
La cassure est detectee sur la premiere bougie post-OR dont le `close` sort de l'OR, en arretant la recherche avant le bar utilise comme close final.

## Data sanity

- MES: fichier `MES_c_0_1m_20260322_135702.parquet`, timezone `America/New_York`, 1,747 sessions RTH, 1,684 sessions avec close proxy >= 15:55.
  Jours eligibles par fenetre: OR 5m: 1,683, OR 15m: 1,681, OR 30m: 1,681, OR 60m: 1,679.
- MNQ: fichier `MNQ_c_0_1m_20260321_094501.parquet`, timezone `America/New_York`, 1,747 sessions RTH, 1,684 sessions avec close proxy >= 15:55.
  Jours eligibles par fenetre: OR 5m: 1,682, OR 15m: 1,681, OR 30m: 1,680, OR 60m: 1,678.
- MGC: fichier `MGC_c_0_1m_20260322_155729.parquet`, timezone `America/New_York`, 3,128 sessions RTH, 1,866 sessions avec close proxy >= 15:55.
  Jours eligibles par fenetre: OR 5m: 1,168, OR 15m: 966, OR 30m: 852, OR 60m: 781.
- M2K: fichier `M2K_c_0_1m_20260322_134808.parquet`, timezone `America/New_York`, 1,747 sessions RTH, 1,684 sessions avec close proxy >= 15:55.
  Jours eligibles par fenetre: OR 5m: 1,679, OR 15m: 1,675, OR 30m: 1,670, OR 60m: 1,661.

## Direct answers

1. Best global hit rate: **MNQ** avec 54.4% sur 6,627 valid breakouts agreges.
2. Best bullish breakout hit rate: **MNQ** avec 59.4% sur 3,451 breakouts up.
3. Best bearish breakout hit rate: **M2K** avec 52.2% sur 3,309 breakouts down.
4. OR window with the best hit rate for each asset:
- M2K: OR 60 min (54.9%, 1,534 valid breakouts).
- MES: OR 30 min (53.8%, 1,676 valid breakouts).
- MGC: OR 60 min (55.1%, 717 valid breakouts).
- MNQ: OR 60 min (56.0%, 1,601 valid breakouts).
5. OR window with the most valid breakouts for each asset:
- M2K: OR 5 min (1,678 valid breakouts, hit rate 53.4%).
- MES: OR 5 min (1,683 valid breakouts, hit rate 53.4%).
- MGC: OR 5 min (1,168 valid breakouts, hit rate 49.3%).
- MNQ: OR 5 min (1,682 valid breakouts, hit rate 52.8%).
6. Most promising asset for a simple directional ORB read: **MNQ** avec l'OR 30 min (54.8% de hit rate, 1,665 breakouts valides, 0.9% de no-breakout).
7. Least noisy asset under this measure: **MNQ** avec 54.4% de hit rate global et une extension mediane de 11.50.
8. Asset with the fewest no-breakout days: **MES** sur l'OR 5 min (0.0% de no-breakout).

## Vue agregee par asset

```text
asset  n_days  n_no_breakout  n_valid_breakouts  n_breakout_up  n_breakout_down  n_same_direction global_hit_rate global_hit_rate_up global_hit_rate_down global_no_breakout_rate  avg_close_extension  median_close_extension
  M2K    6685            156               6529           3220             3309              3529           54.1%              56.0%                52.2%                    2.3%             0.997825                    1.50
  MES    6724             59               6665           3479             3186              3566           53.5%              57.5%                49.2%                    0.9%             1.854314                    2.25
  MGC    3767             86               3681           1880             1801              1931           52.5%              55.2%                49.6%                    2.3%             1.028932                    0.60
  MNQ    6721             94               6627           3451             3176              3603           54.4%              59.4%                48.9%                    1.4%             8.404142                   11.50
```

## Top configurations

```text
asset  or_window_minutes  n_days  n_valid_breakouts hit_rate hit_rate_up hit_rate_down pct_no_breakout avg_close_extension median_close_extension
  MNQ                 60    1678               1601    56.0%       61.3%         49.9%            4.6%                8.14                  11.25
  MGC                 60     781                717    55.1%       58.2%         51.6%            8.2%                1.39                   0.70
  M2K                 60    1661               1534    54.9%       58.2%         51.5%            7.6%                0.94                   1.30
  MNQ                 30    1680               1665    54.8%       60.5%         48.4%            0.9%                9.42                  13.00
  M2K                 30    1670               1648    54.4%       55.4%         53.3%            1.3%                1.14                   1.60
  MNQ                 15    1681               1679    54.0%       58.8%         48.9%            0.1%                7.60                  13.00
  MES                 30    1681               1676    53.8%       57.9%         49.2%            0.3%                1.92                   2.12
  MGC                 30     852                835    53.7%       57.5%         49.6%            2.0%                1.42                   0.80
```

## Charts

- `charts/heatmap_hit_rate.png`
- `charts/heatmap_hit_rate_up.png`
- `charts/heatmap_hit_rate_down.png`
- `charts/heatmap_no_breakout_rate.png`
- `charts/heatmap_avg_close_extension.png`
- `charts/heatmap_valid_breakouts.png`
