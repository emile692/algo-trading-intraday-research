# Revue des baselines canoniques figées

Cette étape transforme les 16 mécanismes représentatifs en 16 spécifications exécutables. Aucun résultat de backtest, classement, Sharpe ou Profit Factor publié n'a servi à choisir une règle ou un paramètre. Aucun backtest n'est lancé par le script de construction.

Le seul benchmark préparé est `MNQ_BUY_AND_HOLD`. Il est défini séparément des stratégies et ne participe ni au choix des mécanismes ni au gel de leurs paramètres.

Le détail de gel et les empreintes SHA-256 sont dans [`frozen_baselines_manifest.csv`](frozen_baselines_manifest.csv).

## Les 16 mécanismes figés

| Mécanisme | Représentant | Famille | Grade | Horizon | Hypothèse économique |
|---|---|---|---:|---|---|
| [`trend_pullback_ma_structure`](frozen_baselines/trend_pullback_ma_structure.json) | [`ema-ribbon-trend`](cards/ema-ribbon-trend.md) | trend_momentum | C | 15min | Persistent order flow can keep returns directionally dependent while temporary profit taking pulls price toward fast moving averages. |
| [`vwap_equilibrium_reversion`](frozen_baselines/vwap_equilibrium_reversion.json) | [`vwap-deviation`](cards/vwap-deviation.md) | mean_reversion | B | 5min | Session VWAP proxies traded equilibrium; liquidity shocks can push price away from it before two-sided flow restores balance. |
| [`rolling_range_breakout`](frozen_baselines/rolling_range_breakout.json) | [`donchian-breakout`](cards/donchian-breakout.md) | breakout_expansion | B | 1H | A range escape can reveal information-driven order imbalance; the shorter exit channel responds when that imbalance reverses. |
| [`atr_trend_flip`](frozen_baselines/atr_trend_flip.json) | [`supertrend`](cards/supertrend.md) | trend_momentum | A | 15min | ATR-scaled bands adapt the reversal threshold to current noise, allowing persistent moves to remain active while filtering smaller counter-moves. |
| [`ma_crossover_trend`](frozen_baselines/ma_crossover_trend.json) | [`dual-ma-crossover`](cards/dual-ma-crossover.md) | trend_momentum | B | 15min | Smoothed price acceleration can proxy persistent order flow, while ADX suppresses low-directionality crossovers. |
| [`session_range_breakout`](frozen_baselines/session_range_breakout.json) | [`opening-range-breakout`](cards/opening-range-breakout.md) | breakout_expansion | B | 5min | The opening range aggregates overnight repricing and initial liquidity; a volume-expanding escape can indicate persistent imbalance. |
| [`compression_breakout`](frozen_baselines/compression_breakout.json) | [`range-compression-breakout`](cards/range-compression-breakout.md) | breakout_expansion | C | 1H | Volatility clusters; unusually narrow balance can precede repricing when resting liquidity is exhausted on one side. |
| [`statistical_zscore_reversion`](frozen_baselines/statistical_zscore_reversion.json) | [`z-score-mean-reversion`](cards/z-score-mean-reversion.md) | mean_reversion | B | 1H | Temporary order imbalance can move price beyond its recent statistical equilibrium before liquidity provision restores it. |
| [`regression_equilibrium_reversion`](frozen_baselines/regression_equilibrium_reversion.json) | [`regression-to-mean`](cards/regression-to-mean.md) | mean_reversion | C | 1H | A local trend can persist while temporary liquidity shocks create residual excursions around that trend, allowing reversion without requiring a flat market. |
| [`volatility_breakout`](frozen_baselines/volatility_breakout.json) | [`volatility-breakout`](cards/volatility-breakout.md) | breakout_expansion | B | 5min | A move large relative to recent daily range can indicate a trend-day information shock rather than ordinary opening noise. |
| [`short_horizon_pullback_reversion`](frozen_baselines/short_horizon_pullback_reversion.json) | [`rsi-2-pullback`](cards/rsi-2-pullback.md) | mean_reversion | C | 15min | Very short-horizon selling or buying pressure can become exhausted while the slower trend remains intact. |
| [`momentum_confirmed_breakout`](frozen_baselines/momentum_confirmed_breakout.json) | [`momentum-breakout`](cards/momentum-breakout.md) | breakout_expansion | C | 1H | Price discovery accompanied by momentum and directional strength can reflect broad participation rather than a transient stop run. |
| [`gap_fade`](frozen_baselines/gap_fade.json) | [`gap-fill`](cards/gap-fill.md) | session_gap | B | 5min | Small overnight imbalances may reflect thin liquidity rather than durable information and can be arbitraged back toward the prior RTH close. |
| [`gap_continuation`](frozen_baselines/gap_continuation.json) | [`gap-and-go`](cards/gap-and-go.md) | session_gap | C | 15min | A large, well-participated gap can encode durable new information that attracts trend-following flow rather than immediate liquidity-driven reversion. |
| [`range_structure_reversion`](frozen_baselines/range_structure_reversion.json) | [`range-midpoint-reversion`](cards/range-midpoint-reversion.md) | mean_reversion | B | 15min | Balanced auction conditions can attract liquidity near established boundaries and pull price back toward the center of accepted value. |
| [`channel_extreme_reversion`](frozen_baselines/channel_extreme_reversion.json) | [`keltner-mean-reversion`](cards/keltner-mean-reversion.md) | mean_reversion | C | 15min | Moves beyond a volatility-scaled envelope can represent temporary exhaustion; the EMA midline proxies the local equilibrium. |

## Conventions communes ajoutées

- Les indicateurs utilisent seulement des barres RTH closes, en heure `America/New_York`.
- Chaque horizon est choisi dans la liste publiée pour rendre la règle mécanique, sans   comparaison de performance; ce choix est ensuite figé dans le JSON.
- Les barres signal sont construites à partir des données MNQ une minute; aucune barre   incomplète ne peut produire un signal.
- Un signal confirmé à la clôture est exécuté à l'ouverture exacte de la minute suivante.   Les ordres stop déjà connus avant la minute suivent leur règle de déclenchement propre.
- Chaque baseline utilise un contrat MNQ, sans pyramiding ni sizing conditionnel.
- La clôture cash régulière est `16:00:00` en heure `America/New_York`. Elle termine la   fenêtre de nouveaux signaux, sans imposer à elle seule une sortie sauf règle spécifique.
- La limite de nouvelle entrée est `15:55:00` par défaut. `gap_fade`   conserve sa limite spécifique à `09:31:00` et `gap_continuation` à `09:45:00`.
- Le hard flatten Topstep est `16:08:00` : aucune position de stratégie ne   peut subsister après cette heure. Entre 16:00 et 16:08, seuls les stops et cibles déjà   définis restent surveillés; aucun nouveau signal ni aucune nouvelle entrée n'est accepté.
- Lors d'une clôture anticipée officielle du CME, la clôture cash devient l'heure publiée,   la limite d'entrée est le minimum entre la règle de la stratégie et 5 minutes avant cette clôture,   et le hard flatten intervient 1 minute avant. La session est rejetée si le calendrier manque.
- Coûts figés: `1.25 USD` par côté et `1` tick adverse par côté; tick MNQ `0.25` point.
- Si stop et cible sont touchés dans la même minute, le stop est prioritaire.
- Les sorties partielles et options facultatives sont désactivées avec un contrat; leur   omission est explicitement marquée comme `research_assumption`.

## Hypothèses spécifiques ajoutées

| Mécanisme | Conventions spécifiques |
|---|---|
| `trend_pullback_ma_structure` | `ema_pullback_confirmation` |
| `vwap_equilibrium_reversion` | `vwap_band_formula`, `vwap_confirmation`, `fixed_vwap_levels_after_entry` |
| `rolling_range_breakout` | `donchian_close_confirmation` |
| `atr_trend_flip` | `supertrend_defaults` |
| `ma_crossover_trend` | `dual_ma_adx_period` |
| `session_range_breakout` | `opening_range_volume_expansion`, `opening_range_symmetric_buffer`, `opening_range_five_minute_bars` |
| `compression_breakout` | `nr7_tie_rule`, `nr7_entry_buffer` |
| `statistical_zscore_reversion` | `zscore_population_sigma`, `zscore_cross_entries` |
| `regression_equilibrium_reversion` | `regression_window`, `regression_dynamic_exit` |
| `volatility_breakout` | `volatility_breakout_atr` |
| `short_horizon_pullback_reversion` | `rsi2_wilder`, `rsi2_cross_entries` |
| `momentum_confirmed_breakout` | `momentum_breakout_close`, `momentum_trail_definition` |
| `gap_fade` | `gap_fade_entry_delay`, `gap_fade_midday` |
| `gap_continuation` | `gap_go_volume_average`, `gap_go_target_anchor` |
| `range_structure_reversion` | `range_midpoint_rsi`, `range_midpoint_fresh_entry` |
| `channel_extreme_reversion` | `keltner_definition`, `keltner_touch`, `keltner_fixed_exit_levels` |

## Ambiguïtés restantes et exécutabilité

Les 16 fichiers ont `unresolved_ambiguities = []`. Les règles absentes des fiches ont été résolues par la convention mécanique la plus simple et restent visibles dans `remaining_assumptions`. Les 16 spécifications sont donc exécutables sans paramètre libre, mais ces conventions devront être conservées telles quelles pendant le tournoi baseline.
Le manifeste contient le SHA-256 de chaque JSON pour détecter toute modification ultérieure.

**Stratégies non exécutables à ce stade : aucune parmi les 16 représentantes.**

## Risques méthodologiques contrôlés

- **Look-ahead** : les rolling highs/lows excluent la barre testée; VWAP, ATR, RSI, ADX,   régression et bandes ne sont mis à jour qu'après clôture de leur barre.
- **High/low non terminé** : les touches intrabar servent seulement aux ordres dont le   niveau était connu avant la minute; les autres entrées attendent la clôture du signal.
- **Same bar** : stop avant cible; les doubles déclenchements directionnels impossibles à   ordonner annulent le setup.
- **Fills impossibles** : pas de fill au close qui crée le signal; absence de la minute   d'exécution attendue = signal ignoré; les gaps sont remplis au prix ouvert défavorable.
- **Données** : toutes les stratégies utilisent OHLCV, timestamp et `session_date`. Aucune   donnée tick, carnet, volume delta ou volume-at-price n'est reconstruite. Le calendrier   officiel CME est requis pour identifier les clôtures anticipées.
- **Série continue** : les changements de contrat et ajustements de roll peuvent affecter   les signaux et le benchmark; la même série gelée doit être utilisée partout.

## Benchmark MNQ_BUY_AND_HOLD

Le benchmark reprend la convention existante du dépôt: dernier close valide de chaque session et courbe normalisée `capital_initial × close_t / close_0`. Il représente une exposition marché continue close-to-close. La clôture cash de 16:00, ou la clôture CME anticipée officielle, sert uniquement à échantillonner le close. Le benchmark n'applique ni limite d'entrée, ni daily flatten, ni hard flatten Topstep, ni coûts d'exécution des stratégies. La fenêtre de comparaison doit être fixée avant le calcul des performances.

Spécification: [`MNQ_BUY_AND_HOLD.json`](frozen_baselines/MNQ_BUY_AND_HOLD.json).

## Reproduction et prochaine étape

```powershell
python strategy_registry/psf_mnq/build_frozen_baselines.py
```

La prochaine étape autorisée par cette préparation est:

```text
frozen specification -> baseline backtest -> robustness -> OOS
```

Ce document et le script de construction ne lancent aucune de ces simulations.
