# Intraday futures — portefeuille actif de neuf sleeves

La référence opérationnelle active depuis le **4 octobre 2026** est un portefeuille equal-contract de neuf sleeves verrouillées : quatre MNQ, deux MGC et trois ensembles MBT. La référence précédente à six sleeves reste conservée comme baseline historique.

## État actif

Mapping actif : **`[1,1,1,1,1,1,1,1,1]`**, une copie entière par sleeve. Référence Topstep bootstrap 60 jours : **45,2375 % de PASS**.

### Bloc MNQ — 4 sleeves

- `orb_vwap_atr`
- `volatility_breakout`
- `compression_breakout`
- `atr_trend_flip`

Allocation interne verrouillée : **constrained ERC**, poids individuels bornés entre 10% et 40%, covariance rolling 63 séances arrêtée à `t-1`.

### Bloc MGC — 2 sleeves

- `volatility_breakout` : `daily_atr_period=30`, `entry_atr_multiple=1.10`
- `gap_continuation` : `minimum_gap_atr_multiple=0.50`, `volume_threshold_multiple=0.60`

Les paramètres sont gelés. Toute observation future doit être prospective et ne doit pas servir à les retuner.

### Bloc MBT — 3 ensembles de votes

- `rolling_range_breakout`
- `trend_pullback_ma_structure`
- `volatility_breakout`

Chaque ensemble conserve toutes les cellules de sa région gelée et discrétise leur position moyenne vers `-1/0/+1` contrat.

## Documentation canonique

- [Huit sleeves actives](docs/ACTIVE_EIGHT_SLEEVES.md)
- [Neuf sleeves — référence du 1er novembre 2026, remplacée](docs/ACTIVE_NINE_SLEEVES.md)
- [Neuf fiches signal client](docs/signal_sheets/README.md)
- [Registre des recherches non promues](docs/RESEARCH_FAILURE_REGISTER_2026.md)
- [Archive externe des résultats](docs/RESULT_ARCHIVE_2026_10_06.md)
- [Six sleeves — baseline historique](docs/ACTIVE_SIX_SLEEVES.md)
- [Clôture des recherches et échecs](docs/RESEARCH_CLOSURE_2026.md)
- [Carte du dépôt](docs/REPOSITORY_MAP.md)
- [Configuration MNQ verrouillée](research/locked_four_sleeve_portfolio/locked_four_sleeve_config.json)
- [Configuration MGC verrouillée](research/mgc_final_two_sleeve_lock_20260926/final_lock_config.json)

## Utilisation

```powershell
uv sync
.venv\Scripts\python.exe -m pytest -p no:cacheprovider tests/test_frozen_baseline_engine.py tests/test_backtester.py tests/test_orb_strategy.py tests/test_locked_nine_sleeve_portfolio.py -q
```

Cette commande est la vérification courte de la chaîne encore active. La suite
historique complète contient volontairement des tests de campagnes et de
notebooks archivés; elle n'est plus la gate de maintenance du portefeuille.

Les secrets Databento ne doivent jamais être stockés dans le dépôt. Les données Parquet locales sont conservées sous `data/` mais restent exclues du suivi Git.

## Politique de recherche

- Ne pas rouvrir les branches fermées à partir des mêmes données.
- Ne pas modifier les paramètres, régions ou règles de vote des neuf sleeves sur la base de la période déjà observée.
- Séparer clairement suivi prospectif, maintenance du moteur et nouvelle recherche.
- Toute nouvelle expérience doit avoir une hypothèse, des paramètres et une règle d'arrêt écrits avant calcul.
