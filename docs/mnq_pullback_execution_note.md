# MNQ pullback retenu - note d'implementation execution

Cette note decrit la jambe `client_pullback_core` retenue dans `notebooks/mnq_orb_pullback_research_client.executed.ipynb`.
Elle vise une implementation dans un repo d'execution, sans dependre des exports de campagne.

Sources de reference dans ce repo:

- `src/analytics/build_mnq_orb_pullback_research_client_notebook.py`: parametrage du notebook client et orchestration du run.
- `src/strategy/volume_climax_pullback_v2.py`: construction des features et du signal.
- `src/engine/volume_climax_pullback_v2_backtester.py`: logique d'entree, sorties, sizing et PnL.
- `src/risk/position_sizing.py`: calcul du nombre de contrats.
- `src/config/settings.py`: specifications MNQ et couts.

## 1. Objet de la strategie

La strategie pullback est une strategie de reversal intraday sur MNQ en barres RTH 1h.

Elle cherche une bougie "climax" de volume:

- si la bougie setup 1h precedente est haussiere, la strategie vend a decouvert a l'ouverture de la bougie suivante;
- si la bougie setup 1h precedente est baissiere, la strategie achete a l'ouverture de la bougie suivante;
- le stop est l'extreme de la bougie setup;
- la cible est a `1.0 * ATR20` depuis le prix d'entree;
- un time stop coupe le trade apres 2 barres de gestion si ni stop ni target n'a ete touche.

Le notebook client compare ensuite cette jambe au signal ORB, puis construit un blend journalier equal weight 50/50. Si le repo d'execution implemente seulement `mnq_pullback`, la partie ORB/blend est optionnelle.

## 2. Parametres retenus

### Univers et donnees

| Parametre | Valeur |
|---|---:|
| Symbole | `MNQ` |
| Source recherche | dernier parquet MNQ 1m, ici `data/processed/parquet/MNQ_c_0_1m_20260321_094501.parquet` |
| Timezone attendue | timestamps locaux US/Eastern dans les donnees traitees |
| Session utilisee | RTH `09:30` a `16:00` |
| Timeframe signal | `1h` |
| Resampling | `1h`, `label="left"`, `closed="left"`, `origin="start_day"`, `offset="30min"` |
| Capital de reference | `50_000 USD` |
| Split recherche | `70%` in-sample, `30%` out-of-sample |

### Specifications MNQ et couts

| Parametre | Valeur |
|---|---:|
| Tick size | `0.25` point |
| Tick value | `0.50 USD` |
| Point value | `2.00 USD` |
| Commission | `1.25 USD` par cote et par contrat |
| Slippage | `1` tick par entree/sortie |
| Profil execution | `repo_realistic` |

### Signal pullback

| Parametre | Valeur | Role |
|---|---:|---|
| `name` | `client_pullback_core` | Nom de la variante retenue |
| `family` | `dynamic_exit` | Famille de recherche |
| `timeframe` | `1h` | Barres de signal |
| `volume_quantile` | `0.95` | Volume setup > quantile 95% historique |
| `volume_lookback` | `50` | Lookback en barres 1h pour le seuil de volume |
| `min_body_fraction` | `0.50` | Corps minimum: `abs(close-open)/(high-low)` |
| `min_range_atr` | `1.20` | Range setup minimum en multiple d'ATR20 |
| `trend_ema_window` | `None` | Pas de filtre tendance |
| `ema_slope_threshold` | `None` | Pas de filtre pente EMA |
| `atr_percentile_low/high` | `None / None` | Pas de filtre percentile ATR |
| `compression_ratio_max` | `None` | Pas de filtre compression |
| `entry_mode` | `next_open` | Entree a l'ouverture de la barre suivant le setup |
| `pullback_fraction` | `None` | Desactive, car pas d'entree limite |
| `confirmation_window` | `None` | Desactive, car pas d'entree confirmation |
| `session_overlay` | `all_rth` | La strategie travaille sur toute la RTH resamplee |

### Sorties

| Parametre | Valeur | Role |
|---|---:|---|
| `exit_mode` | `atr_fraction` | Cible en fraction/multiple d'ATR |
| `atr_target_multiple` | `1.00` | Target = entree +/- `1.0 * ATR20 setup` |
| `time_stop_bars` | `2` | Sortie au close apres 2 barres de gestion |
| `trailing_atr_multiple` | `0.50` | Parametre present mais non utilise par `atr_fraction` |
| `rr_target` | `1.0` | Parametre present mais non utilise par `atr_fraction` |

### Sizing

| Parametre | Valeur |
|---|---:|
| Mode | `RiskPercentPositionSizing` |
| `initial_capital_usd` | `50_000` |
| `risk_pct` | `0.0025` soit `0.25%` du capital initial constant |
| `max_contracts` | `6` |
| `skip_trade_if_too_small` | `True` |
| `compound_realized_pnl` | `False` |

### Portefeuille client

| Parametre | Valeur |
|---|---:|
| `ORB_WEIGHT` | `0.50` |
| `PULLBACK_WEIGHT` | `0.50` |
| `ORB_LEVERAGE` | `1.00` |
| `PULLBACK_LEVERAGE` | `1.00` |
| `BLEND_LEVERAGE` | `1.00` |

## 3. Pipeline exact a implementer

### Etape 1 - Charger et filtrer les donnees

1. Charger les OHLCV MNQ 1 minute nettoyes.
2. Conserver la session RTH, timestamps dont l'heure est entre `09:30` et `16:00`.
3. Resampler en barres 1h avec ancrage a `09:30`:
   - open = premier open;
   - high = max high;
   - low = min low;
   - close = dernier close;
   - volume = somme des volumes.
4. Supprimer les barres sans OHLC complet.
5. Ajouter `session_date = date(timestamp)`.

Dans la recherche executee, ce pipeline donne:

| Item | Valeur |
|---|---:|
| Barres 1h | `12_035` |
| Sessions totales | `1_747` |
| Sessions IS | `1_222` |
| Sessions OOS | `525` |

### Etape 2 - Construire les features sur les barres 1h

Pour chaque barre 1h:

```text
prev_close = close.shift(1)
true_range = max(high-low, abs(high-prev_close), abs(low-prev_close))
atr_20 = mean(true_range, 20 barres)
bar_range = high - low
body = abs(close - open)
body_fraction = body / bar_range
range_atr = bar_range / atr_20
```

Le code calcule aussi `atr_5`, `atr_50`, `ema20`, `ema50`, `session_vwap`, `atr_percentile_100` et `atr_ratio_5_20`, mais la variante retenue ne les utilise pas comme filtres. `session_vwap` n'est pas utilise par `exit_mode="atr_fraction"`.

### Etape 3 - Detecter la bougie setup

Le signal est projete sur la barre courante `t`, mais il se base uniquement sur la bougie precedente `t-1`.

Important pour reproduire exactement la recherche:

```text
volume_threshold_hist[t] = quantile(volume[t-50 ... t-1], 0.95)
setup = barre t-1
```

Le seuil de volume est donc connu a la cloture de la bougie setup et inclut cette bougie setup dans la fenetre des 50 volumes termines. Ne pas remplacer par un seuil calcule seulement sur `t-51 ... t-2`, sinon les signaux changeront.

Une bougie setup est valide si toutes les conditions suivantes sont vraies:

```text
volume[t-1] > volume_threshold_hist[t]
body_fraction[t-1] >= 0.50
range_atr[t-1] >= 1.20
```

Direction:

```text
si close[t-1] > open[t-1]  => signal[t] = -1  # short apres climax haussier
si close[t-1] < open[t-1]  => signal[t] = +1  # long apres climax baissier
sinon                      => signal[t] = 0
```

Aucun filtre supplementaire n'est actif dans la variante retenue. Dans le notebook execute, `raw_signal_count = filtered_signal_count = 324`.

### Etape 4 - Entrer en position

Mode d'entree: `next_open`.

Quand `signal[t] != 0`, le trade est tente a l'ouverture de la barre `t`.

Prix d'entree brut:

```text
raw_entry_price = open[t]
```

Prix d'entree apres slippage:

```text
long  : entry_price = raw_entry_price + 1 tick = raw_entry_price + 0.25
short : entry_price = raw_entry_price - 1 tick = raw_entry_price - 0.25
```

Contraintes de position:

- une seule position ouverte a la fois;
- pas de re-entree sur la meme barre apres une sortie;
- pas de limite explicite "un trade par jour";
- une nouvelle entree peut arriver plus tard si la position precedente est fermee.

Nuance backtest a conserver si l'objectif est une parite stricte avec le notebook: les stops/targets ne sont pas testes sur la barre d'entree elle-meme. La gestion commence a partir de la barre suivante.

### Etape 5 - Placer le stop initial

Le stop vient de l'extreme de la bougie setup `t-1`:

```text
long  : initial_stop_price = low[t-1]
short : initial_stop_price = high[t-1]
```

Le trade est saute si:

- le stop est manquant;
- la distance au stop est nulle ou negative apres slippage.

Distance de risque:

```text
risk_points = (entry_price - stop_price) * direction
```

avec `direction = +1` pour long, `direction = -1` pour short.

### Etape 6 - Calculer le target

La variante retenue utilise `exit_mode="atr_fraction"`.

ATR de reference:

```text
reference_atr = atr_20[t-1]
```

Target:

```text
long  : target_price = entry_price + 1.00 * reference_atr
short : target_price = entry_price - 1.00 * reference_atr
```

Le trade est saute si `reference_atr` est invalide ou <= 0.

### Etape 7 - Calculer la taille de position

Capital de sizing:

```text
capital_before_trade_usd = 50_000 USD
```

Le PnL realise continue d'alimenter les courbes d'equity et les metriques, mais il ne modifie pas le capital utilise pour dimensionner le trade suivant.

Budget de risque:

```text
risk_budget_usd = capital_before_trade_usd * 0.0025
```

Risque par contrat:

```text
risk_per_contract_usd = abs(entry_price - initial_stop_price) * 2.0
```

Le sizing ne rajoute pas les commissions au risque par contrat. Les commissions sont deduites ensuite du PnL.

Contrats:

```text
contracts_raw = risk_budget_usd / risk_per_contract_usd
contracts = floor(contracts_raw)
contracts = min(contracts, 6)
```

Si `contracts < 1` et `skip_trade_if_too_small=True`, le trade est saute avec raison `contracts_below_one`.

### Etape 8 - Gerer la sortie

A chaque nouvelle barre 1h tant que le trade est ouvert:

1. Incrementer `bars_held`.
2. Tester stop et target sur le high/low de la barre.
3. Si stop et target sont touches dans la meme barre, choisir le stop: `stop_ambiguous_first`.
4. Sinon sortir au stop si touche.
5. Sinon sortir au target si touche.
6. Sinon, si `bars_held >= 2`, sortir au close de la barre: `time_stop`.
7. Sinon, si c'est la derniere barre de la session et que le trade etait deja ouvert au debut de cette barre, sortir au close: `eod_flat`.

Prix de sortie apres slippage:

```text
long  : exit_fill = raw_exit_price - 0.25
short : exit_fill = raw_exit_price + 0.25
```

Le backtester est conservateur sur les barres ambigues: stop avant target.

### Etape 9 - Calculer le PnL

PnL brut:

```text
pnl_points = (exit_price - entry_price) * direction
pnl_usd = pnl_points * 2.0 * contracts
```

Frais:

```text
fees = 2 * 1.25 * contracts
```

PnL net:

```text
net_pnl_usd = pnl_usd - fees
```

Apres chaque cloture, le PnL est comptabilise dans les resultats journaliers et l'equity curve. En revanche, le sizing du trade suivant reste calcule sur `50_000 USD`.

## 4. Resultats de controle du notebook execute

Ces chiffres servent de smoke test si le repo d'execution rejoue le meme historique et les memes hypotheses.

### Pullback standalone

| Scope | Net PnL | Return | CAGR | Sharpe | Sortino | Max DD | Trades | Win rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | `35_688.10 USD` | `71.376%` | `8.154%` | `1.804` | `2.995` | `1_526.23 USD` | `288` | `35.1%` |
| IS | `25_555.60 USD` | `51.111%` | `8.977%` | `1.871` | `3.044` | `1_526.23 USD` | `203` | `36.5%` |
| OOS rebased | `10_132.50 USD` | `20.265%` | `9.325%` | `1.640` | `2.898` | `776.60 USD` | `85` | `31.8%` |

### Common sample avec ORB/blend

Sample commun: `2019-05-06` -> `2026-03-19`.

Debut OOS commun: `2024-02-28`.

| Strategie | Scope | Net PnL | Sharpe | Max DD |
|---|---|---:|---:|---:|
| Pullback | full_common | `35_688.10 USD` | `1.804` | `1_526.23 USD` |
| Pullback | oos_common | `6_705.33 USD` | `1.644` | `513.93 USD` |
| Blend 50/50 ORB/Pullback | oos_common | `5_733.17 USD` | `2.359` | `387.52 USD` |

Correlation journaliere ORB/Pullback:

- full: `-0.031`;
- OOS: `-0.024`.

## 5. Grilles de robustesse du notebook

Le notebook retient `volume_quantile=0.95`, `min_body_fraction=0.50`, `min_range_atr=1.20`.

Grille alpha testee:

```text
volume_quantile      = (0.95, 0.975)
min_body_fraction    = (0.50, 0.60)
min_range_atr        = (1.20, 1.50)
```

Top OOS de cette grille alpha: `0.95 / 0.50 / 1.20`, identique a la variante client.

Grille sizing testee:

```text
risk_pct      = (0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040)
max_contracts = (2, 3, 4, 5, 6)
```

Le sizing retenu par le notebook client est `risk_pct=0.0025`, `max_contracts=6`. La grille montre d'autres compromis Sharpe/PnL, mais elle ne remplace pas le parametrage client retenu.

## 6. Checklist implementation execution

Pour coller a la recherche:

1. Construire les barres 1h RTH avec le meme ancrage `09:30`.
2. Calculer les features uniquement avec les barres terminees.
3. A la cloture de chaque barre 1h, evaluer si cette barre devient un setup valide.
4. Si setup valide, preparer l'ordre marche pour l'ouverture de la barre 1h suivante dans le sens oppose a la bougie setup.
5. Appliquer le slippage adverse d'un tick a l'entree et a la sortie dans les simulations.
6. Placer stop a l'extreme setup et target a `1.0 * ATR20 setup`.
7. Sizer depuis le risque au stop, `0.25%` du capital initial constant, `floor`, cap `6` contrats, skip si < 1 contrat.
8. Gerer stop, target, time stop 2 barres et EOD flat avec la priorite exacte: stop ambigu avant target.
9. Logger au minimum: session date, setup time, entry time, direction, entry, stop, target, ATR setup, volume threshold, quantity, risk budget, contracts_raw, exit time, exit reason, gross/net PnL, fees.
10. Ne pas ajouter de filtre VWAP, trend EMA, percentile ATR, compression ou ORB a cette jambe pullback: ils sont desactives dans la variante retenue.

## 7. Points d'attention

- La strategie est contrarienne: short apres climax haussier, long apres climax baissier.
- Le seuil de volume utilise la fenetre de 50 barres terminees jusqu'a la bougie setup incluse.
- L'ATR de target et le range ATR de qualite sont ceux de la bougie setup, donc disponibles a sa cloture.
- `rr_target` et `trailing_atr_multiple` sont presents dans la config mais sans effet avec `exit_mode="atr_fraction"`.
- Le backtest ne teste pas stop/target sur la barre d'entree elle-meme. Si le moteur d'execution live le fait, il ne sera pas strictement comparable a la recherche.
- Le sizing ne compose pas le PnL realise: le budget de risque reste base sur `50_000 USD`.
- Le sizing ne compte pas les commissions dans le risque au stop; elles sont deduites seulement dans le PnL net.
- Les resultats du notebook incluent slippage et commissions du profil `repo_realistic`.
