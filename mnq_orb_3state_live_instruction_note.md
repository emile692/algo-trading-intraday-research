# Note d'instruction - implementation du 3-state live dans `algo-trading-intraday-execution`

## Objectif

Implementer dans le repo `algo-trading-intraday-execution` un **overlay de sizing 3-state** pour `mnq_orb`, sans changer la logique d'entree/sortie ORB deja en production locale.

Le changement attendu est :

- le **signal ORB reste identique**
- les **niveaux stop/target restent identiques**
- la **quantity** devient modulee par un bucket 3-state base sur `realized_vol_ratio_15_60`
- le comportement doit rester **safe en live**, **lisible au monitor**, et **teste**

## Source de verite research

Le repo d'execution actuel pointe vers une version ORB sans 3-state dans [configs/strategies/mnq_orb.yaml](d:/Business/Trading/VSCODE/algo-trading-intraday-execution/configs/strategies/mnq_orb.yaml).

La source research a porter est la variante client retenue, pas le mapping brut initial :

- notebook client retenu : [notebooks/orb_MNQ_sizing_3state_client.executed.ipynb](d:/Business/Trading/VSCODE/algo-trading-intraday-research/notebooks/orb_MNQ_sizing_3state_client.executed.ipynb)
- builder associe : [src/analytics/build_mnq_orb_sizing_3state_client_notebook.py](d:/Business/Trading/VSCODE/algo-trading-intraday-research/src/analytics/build_mnq_orb_sizing_3state_client_notebook.py)
- stress test high bucket : [src/analytics/mnq_orb_3state_high_bucket_stress.py](d:/Business/Trading/VSCODE/algo-trading-intraday-research/src/analytics/mnq_orb_3state_high_bucket_stress.py)
- rapport de stress : [export/mnq_orb_3state_high_bucket_stress_20260530_152400/high_bucket_stress_report.md](d:/Business/Trading/VSCODE/algo-trading-intraday-research/export/mnq_orb_3state_high_bucket_stress_20260530_152400/high_bucket_stress_report.md)

Important :

- l'export de campagne `sizing_3state_realized_vol_ratio_15_60` porte par defaut `high -> 0.75x`
- mais la **version live-oriented retenue** apres stress test est :
  - `low = 0.50x`
  - `mid = 1.00x`
  - `high = 0.25x`

Le rapport du 30 mai 2026 recommande explicitement `high_0p25`.

## Ce qu'il ne faut pas changer

Ne pas modifier :

- la detection ORB elle-meme
- la confirmation VWAP
- le timing `signal bar close -> next bar open`
- la construction du stop sous l'opening range
- la construction du target a `initial_risk * target_multiple`
- le routeur d'ordres live-safe
- la persistance broker/state existante, sauf si necessaire pour l'observabilite

En pratique, le changement doit rester **localise au signal ORB + diagnostics + config + tests**.

## Specification fonctionnelle exacte

### 1. Feature 3-state a calculer

Le feature a utiliser est :

- `realized_vol_ratio_15_60 = vol_std_15 / vol_std_60`

avec :

- `vol_std_15 = rolling std des close returns sur 15 barres`
- `vol_std_60 = rolling std des close returns sur 60 barres`
- return unitaire : `close.pct_change()`

La logique research correspond a :

- calcul sur les **barres minute continues**
- pas de reset par session
- pas de lookahead
- bucketisation appliquee sur la valeur disponible **au moment de la barre signal**

Reference research : `add_rolling_std(..., window=15|60)` dans [src/features/volatility.py](d:/Business/Trading/VSCODE/algo-trading-intraday-research/src/features/volatility.py) puis `realized_vol_ratio_15_60` dans [src/analytics/mnq_orb_regime_filter_sizing_campaign.py](d:/Business/Trading/VSCODE/algo-trading-intraday-research/src/analytics/mnq_orb_regime_filter_sizing_campaign.py).

### 2. Bucketisation a porter en live

Ne pas utiliser les bornes min/max observees comme bornes fermees. Pour etre fidele a `pd.cut(..., include_lowest=True)` cote research, utiliser des seuils ouverts aux extremes :

- `low` si `value <= 0.9429454367121718`
- `mid` si `0.9429454367121718 < value <= 1.1407799880685539`
- `high` si `value > 1.1407799880685539`

Puis appliquer les multiplicateurs retenus :

- `low -> 0.50`
- `mid -> 1.00`
- `high -> 0.25`

### 3. Moment exact d'evaluation du bucket

Le bucket doit etre evalue sur la **barre de breakout qui declenche `_pending_entry`**, pas sur la barre d'entree suivante.

Donc :

- quand `consume_bar()` detecte un breakout valide
- calculer le `realized_vol_ratio_15_60` courant
- resoudre `bucket_label` et `risk_multiplier`
- stocker ces informations dans `PendingEntrySignal`
- reutiliser ces valeurs inchangees dans `_build_trade_plan()` sur la barre suivante

Ne pas recalculer le bucket au moment de l'entree, sinon le sizing pourra diverger entre signal bar et entry bar.

### 4. Effet attendu sur le sizing

Conserver la logique actuelle de sizing dynamique, mais remplacer :

- `risk_budget = nominal_account_size_usd * base_risk_per_trade_pct / 100`

par :

- `effective_risk_per_trade_pct = base_risk_per_trade_pct * risk_multiplier`
- `risk_budget = nominal_account_size_usd * effective_risk_per_trade_pct / 100`

Le calcul du `risk_per_contract` ne change pas.

Avec la config execution actuelle :

- `nominal_account_size_usd = 150000`
- `risk_per_trade_pct = 0.50`

le budget nominal actuel vaut deja `750 USD`, ce qui est coherent avec le baseline research `50000 * 1.5% = 750 USD`.

Les budgets live obtenus avec le 3-state retenu seront donc :

- `low`: `375 USD`
- `mid`: `750 USD`
- `high`: `187.5 USD`

## Plan d'implementation par fichier

## A. Modifier la config strategie

Fichier : [configs/strategies/mnq_orb.yaml](d:/Business/Trading/VSCODE/algo-trading-intraday-execution/configs/strategies/mnq_orb.yaml)

Ajouter un bloc explicite, par exemple :

```yaml
sizing_overlay:
  three_state:
    enabled: true
    feature_name: realized_vol_ratio_15_60
    short_window: 15
    long_window: 60
    fallback_mode: neutral
    fallback_multiplier: 1.0
    buckets:
      low:
        upper_bound: 0.9429454367121718
        risk_multiplier: 0.50
      mid:
        upper_bound: 1.1407799880685539
        risk_multiplier: 1.00
      high:
        risk_multiplier: 0.25
```

Contraintes :

- garder `risk.risk_per_trade_pct` comme base nominale
- ne pas remplacer le bloc `risk` actuel
- ne pas toucher au stop/target config
- le bloc doit rester optionnel pour que le signal fonctionne aussi si `three_state.enabled=false`

## B. Etendre les parametres du signal

Fichier : [src/intraday_execution/signals/opening_range_breakout.py](d:/Business/Trading/VSCODE/algo-trading-intraday-execution/src/intraday_execution/signals/opening_range_breakout.py)

1. Etendre `OpeningRangeBreakoutParams` avec :

- `three_state_enabled: bool`
- `three_state_feature_name: str | None`
- `three_state_short_window: int`
- `three_state_long_window: int`
- `three_state_fallback_mode: str`
- `three_state_fallback_multiplier: float`
- `three_state_low_upper_bound: float | None`
- `three_state_mid_upper_bound: float | None`
- `three_state_low_multiplier: float`
- `three_state_mid_multiplier: float`
- `three_state_high_multiplier: float`

2. Etendre `PendingEntrySignal` pour figer le contexte du signal :

- `signal_timestamp`
- `stop_price`
- `feature_value: float | None`
- `bucket_label: str | None`
- `risk_multiplier: float`
- `effective_risk_per_trade_pct: float`
- `fallback_active: bool = False`
- `fallback_reason: str | None = None`

3. Dans `__init__`, ajouter l'etat interne necessaire pour le calcul causal :

- `self._previous_close: float | None`
- une structure pour les returns roulants
- ou plus simplement deux `deque` de returns, une 15 et une 60

Je recommande `deque` + recalcul `statistics.pstdev` ou `numpy.std(ddof=1)` a chaque barre, car c'est simple et robuste vu la taille minuscule des fenetres.

## C. Calculer le feature en temps reel dans le signal

Toujours dans [src/intraday_execution/signals/opening_range_breakout.py](d:/Business/Trading/VSCODE/algo-trading-intraday-execution/src/intraday_execution/signals/opening_range_breakout.py)

Ajouter une logique interne du type :

1. Au debut de `consume_bar()`, avant la logique ORB :

- calculer `return_t = bar.close / previous_close - 1.0` si `previous_close` existe et est non nul
- pousser ce return dans les fenetres 15 et 60
- mettre a jour `previous_close = bar.close`
- produire :
  - `current_vol_std_15`
  - `current_vol_std_60`
  - `current_realized_vol_ratio_15_60`

2. Ajouter un helper dedie, par exemple :

- `_current_three_state_context() -> tuple[feature_value, bucket_label, risk_multiplier, fallback_active, fallback_reason]`

3. Semantique de fallback recommandee :

- si overlay desactive : multiplier `1.0`, bucket `"disabled"`
- si `vol_std_60` indisponible, nul ou non fini : multiplier `1.0`, bucket `"fallback_neutral"`, `fallback_active=true`
- si `feature_value` non fini : meme fallback neutre

Pourquoi ce choix :

- le repo d'execution doit rester operable meme si l'historique charge au demarrage est un peu court
- mais l'operateur doit voir explicitement que le 3-state n'a pas ete applique nominalement

Ne pas fallback a `0.0` par defaut en live.

## D. Appliquer le multiplicateur dans `_resolve_quantity`

Dans [src/intraday_execution/signals/opening_range_breakout.py](d:/Business/Trading/VSCODE/algo-trading-intraday-execution/src/intraday_execution/signals/opening_range_breakout.py)

Faire evoluer `_resolve_quantity()` pour accepter le multiplicateur effectivement gele a la barre signal, par exemple :

```python
def _resolve_quantity(self, entry_price: float, risk_multiplier: float = 1.0) -> int:
```

Puis :

- calculer `effective_risk_pct = self.params.risk_per_trade_pct * risk_multiplier`
- utiliser `effective_risk_pct` au lieu de `self.params.risk_per_trade_pct`

Ensuite, dans `_build_trade_plan(bar)`, lire le `risk_multiplier` depuis `pending`.

Important :

- toutes les protections doivent garder la meme `quantity` que l'entree
- le `TradePlan` n'a pas besoin d'etre modifie structurellement si la quantity finale est correcte

## E. Enrichir les diagnostics

Toujours dans [src/intraday_execution/signals/opening_range_breakout.py](d:/Business/Trading/VSCODE/algo-trading-intraday-execution/src/intraday_execution/signals/opening_range_breakout.py)

Etendre `diagnostics_snapshot()` avec des champs lisibles :

- `three_state_enabled`
- `three_state_feature_name`
- `three_state_short_window`
- `three_state_long_window`
- `three_state_vol_std_15`
- `three_state_vol_std_60`
- `three_state_feature_value`
- `three_state_bucket_label`
- `three_state_risk_multiplier`
- `three_state_effective_risk_per_trade_pct`
- `three_state_fallback_active`
- `three_state_fallback_reason`

Et si un `pending_plan` ou `pending_entry` existe, les diagnostics doivent refleter **le contexte gele du signal** et non pas seulement l'etat courant des returns.

## F. Exposer le 3-state dans le monitor et les sorties CLI

Fichiers concernes :

- [src/intraday_execution/monitoring/orb_monitor.py](d:/Business/Trading/VSCODE/algo-trading-intraday-execution/src/intraday_execution/monitoring/orb_monitor.py)
- [src/intraday_execution/cli.py](d:/Business/Trading/VSCODE/algo-trading-intraday-execution/src/intraday_execution/cli.py)

A faire :

1. Ajouter dans `snapshot["orb"]` :

- `three_state_feature_value`
- `three_state_bucket_label`
- `three_state_risk_multiplier`
- `three_state_effective_risk_per_trade_pct`
- `three_state_fallback_active`

2. Dans `render_orb_monitor()` afficher une section compacte du type :

- `3-state feature`
- `3-state bucket`
- `3-state multiplier`
- `effective risk %`
- `fallback`

3. Dans `signal-report`, s'assurer que `signal_diagnostics` contient ces champs.

4. Si le replay scene est encore utilise operationnellement, afficher au moins bucket + multiplier.

## G. Ne pas modifier le routeur live

Ne pas toucher a :

- [src/intraday_execution/execution/order_router.py](d:/Business/Trading/VSCODE/algo-trading-intraday-execution/src/intraday_execution/execution/order_router.py)
- [src/intraday_execution/risk/prop_guard.py](d:/Business/Trading/VSCODE/algo-trading-intraday-execution/src/intraday_execution/risk/prop_guard.py)

sauf si un test demontre que la nouvelle quantity necessite un ajustement de garde.

Raison :

- le routeur live-safe gere deja correctement le couple `entry + stop + target`
- le 3-state agit en amont, au niveau de la quantity du `TradePlan`

## H. Attention operationnelle sur les profils live

Point important :

- [configs/app.topstepx.orb.live_smoke.yaml](d:/Business/Trading/VSCODE/algo-trading-intraday-execution/configs/app.topstepx.orb.live_smoke.yaml) force actuellement :
  - `live_smoke.max_entry_quantity = 1`
  - `risk.max_position = 1`

Consequence :

- meme si le 3-state est correctement implemente, le **smoke live restera plafonne a 1 contrat**
- donc l'effet reel du multiplier ne sera pas visible tant que ce profil reste actif

Instruction concrete :

- implementer le 3-state dans la strategie partagee
- conserver `live_smoke` inchange pour la premiere validation securite
- puis prevoir un profil live non-smoke separe si vous voulez observer le sizing reel au-dela de `1`

Ne pas casser le smoke pour gagner du sizing.

## Tests a ajouter ou modifier

### 1. `tests/test_opening_range_breakout.py`

Ajouter des tests qui couvrent :

- calcul correct du bucket `low`
- calcul correct du bucket `mid`
- calcul correct du bucket `high`
- application correcte du multiplicateur sur la quantity
- figer le bucket au moment du signal, meme si la barre suivante changerait de bucket
- fallback neutre quand l'historique ne permet pas encore `vol_std_60`

Exemples attendus :

- a setup identique, quantity `mid` > quantity `low`
- a setup identique, quantity `mid` > quantity `high`
- si `mid=1.0`, la quantity doit reproduire le sizing nominal actuel

### 2. `tests/test_signal_flow.py`

Ajouter un test end-to-end paper :

- config mock avec overlay 3-state active
- verifier que l'entree et les deux protections portent la quantity modulee
- verifier que le state/store recoit bien cette quantity

### 3. `tests/test_orb_monitor.py`

Ajouter un test snapshot :

- le monitor doit renvoyer `three_state_bucket_label`
- `three_state_risk_multiplier`
- et l'information de fallback si applicable

### 4. `tests/test_cli_state_ops.py`

Etendre le test `signal-report` pour verifier que les nouveaux diagnostics 3-state apparaissent dans le JSON.

## Criteres d'acceptation

Le travail est fini seulement si tout ceci est vrai :

1. `mnq_orb` fonctionne encore sans regression quand `three_state.enabled=false`.
2. `mnq_orb` en mode paper produit une quantity reduite en bucket `low` et `high`.
3. le bucket `high` utilise bien `0.25x`, pas `0.75x`.
4. le bucket est determine sur la barre signal, pas la barre d'entree.
5. `signal-report` expose la feature, le bucket et le multiplier.
6. `orb-monitor` expose la feature, le bucket et le multiplier.
7. les tests unitaires et d'integration passent.

## Recommandation de mise en oeuvre

Ordre conseille :

1. ajouter le bloc config 3-state dans `mnq_orb.yaml`
2. etendre `OpeningRangeBreakoutParams` et `PendingEntrySignal`
3. calculer le feature rolling dans `consume_bar()`
4. appliquer le multiplier dans `_resolve_quantity()`
5. exposer les diagnostics
6. mettre a jour `orb_monitor` et `signal-report`
7. ajouter les tests
8. valider d'abord sur `app.yaml` mock, puis sur `app.topstepx.orb.paper.yaml`, puis seulement ensuite sur le profil live smoke

## Formulation courte a donner a Codex dans le repo execution

Implemente dans `mnq_orb` un overlay de sizing 3-state live base sur `realized_vol_ratio_15_60 = rolling_std(close.pct_change(), 15) / rolling_std(close.pct_change(), 60)`, calcule causalement sur les barres minute continues. Gele le bucket au moment de la barre signal ORB, pas a la barre d'entree. Utilise les seuils `<= 0.9429454367121718 = low`, `<= 1.1407799880685539 = mid`, sinon `high`, avec multiplicateurs retenus `low=0.50`, `mid=1.00`, `high=0.25`. Applique ce multiplicateur uniquement au `risk_per_trade_pct` utilise pour resoudre la quantity, sans changer l'entree ORB, le stop ni le target. Expose bucket/multiplier/feature/fallback dans `diagnostics_snapshot`, `signal-report` et `orb-monitor`. Conserve le routeur live-safe inchange et ajoute les tests unitaires/integration/monitor correspondants.
