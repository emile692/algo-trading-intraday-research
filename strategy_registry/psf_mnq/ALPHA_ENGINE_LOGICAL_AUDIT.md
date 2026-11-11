# Revue logique des 43 moteurs d'alpha provisoires

Cette revue utilise uniquement les règles des fiches. Aucun classement, Profit
Factor, Sharpe ou résultat de backtest PineScriptForge ou local n'intervient.

Ici, `alpha_engine` signifie **hypothèse de stratégie autonome à tester**. Le
terme ne constitue pas une affirmation de rentabilité.

## Tests appliqués

1. **Autonomie** : direction, entrée, sortie et invalidation permettent de
   produire une position mécanique.
2. **Hypothèse identifiable** : la raison d'être tient en une phrase qui ne
   dépend pas d'une valeur de paramètre.
3. **Unicité** : les variantes de mesure d'un même phénomène partagent un
   cluster et un représentant.
4. **Complexité** : `A` représente 1–2 paramètres structurants, `B` 3–4, et
   `C` au moins 5 paramètres ou plusieurs conditions imbriquées.
5. **Compatibilité MNQ intraday** : les règles sont exprimables avec les
   données disponibles, sans position overnight. Une stratégie marquée
   `conditional_session_flatten` devra recevoir la même sortie de fin de
   session avant tout backtest.

## Résultat

| Statut | Nombre | Effet |
|---|---:|---|
| `alpha_representative` | 16 | Hypothèse autonome retenue comme représentant |
| `alpha_variant` | 19 | Variante autonome conservée dans le même cluster |
| `demote_trigger` | 3 | Conservée comme déclencheur |
| `demote_filter` | 2 | Conservée comme filtre |
| `merge_duplicate` | 2 | Fusionnée avec la ligne canonique |
| `reject_logic` | 1 | Règles insuffisantes sans invention |
| **Total** | **43** | |

La passe produit donc :

```text
43 lignes provisoires
→ 35 moteurs autonomes uniques
→ 16 mécanismes représentés
```

Les 19 variantes ne comptent pas comme 19 hypothèses économiques
indépendantes. Elles restent disponibles pour une comparaison contrôlée au
sein de leur cluster.

## Les 16 mécanismes et leurs représentants

| Mécanisme | Représentant |
|---|---|
| `trend_pullback_ma_structure` | `ema-ribbon-trend` |
| `atr_trend_flip` | `supertrend` |
| `ma_crossover_trend` | `dual-ma-crossover` |
| `rolling_range_breakout` | `donchian-breakout` |
| `session_range_breakout` | `opening-range-breakout` |
| `compression_breakout` | `range-compression-breakout` |
| `volatility_breakout` | `volatility-breakout` |
| `momentum_confirmed_breakout` | `momentum-breakout` |
| `vwap_equilibrium_reversion` | `vwap-deviation` |
| `statistical_zscore_reversion` | `z-score-mean-reversion` |
| `regression_equilibrium_reversion` | `regression-to-mean` |
| `channel_extreme_reversion` | `keltner-mean-reversion` |
| `short_horizon_pullback_reversion` | `rsi-2-pullback` |
| `range_structure_reversion` | `range-midpoint-reversion` |
| `gap_fade` | `gap-fill` |
| `gap_continuation` | `gap-and-go` |

## Décisions particulières

- `z-score-reversion` fusionne avec `z-score-mean-reversion`.
- `rainbow-moving-average` fusionne avec `rainbow-ma`, elle-même variante de
  l'EMA Ribbon.
- `Bollinger Band Walk` devient un déclencheur d'épuisement et passe dans la
  famille revue `reversal_exhaustion`.
- `Range Midpoint Reversion` passe de `breakout_expansion` à
  `mean_reversion`.
- `HTF Trend + LTF Entry` devient un filtre : les niveaux et pullbacks LTF ne
  sont pas définis mécaniquement.
- `Overnight Gap Strategy` devient un routeur de régime entre `Gap Fill` et
  `Gap & Go`.
- `Kalman Filter Trend` est rejeté dans sa forme publiée : modèle d'état,
  bruits du filtre et seuil de pente ne sont pas spécifiés.
- `VWAP Scalp` est conservé comme trigger seulement; aucune donnée tick ou
  Volume Delta ne sera reconstruite à partir des barres OHLCV.

## Fichier détaillé

[`alpha_engine_logical_audit.csv`](alpha_engine_logical_audit.csv) contient les
43 décisions, les cinq tests, le grade de complexité, le cluster, le
représentant et la justification ligne par ligne.

[`alpha_engine_mechanisms.csv`](alpha_engine_mechanisms.csv) contient les 16
clusters retenus, leur représentant et la liste de leurs variantes autonomes.

L'étape suivante est la [revue des spécifications canoniques figées](FROZEN_BASELINES_REVIEW.md).
Elle transforme chacun des 16 représentants en règles exécutables sans lancer
de backtest.

Reproduction :

```powershell
uv run python strategy_registry/psf_mnq/logical_audit_alpha_engines.py
```
