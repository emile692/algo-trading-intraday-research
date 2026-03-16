# NQ Intraday Research & Backtesting Framework

Base de projet Python (3.10) modulaire pour la recherche intraday sur futures NQ, pilotée depuis un notebook central.

## Structure

- `notebooks/research_main.ipynb`: point d’entrée recherche (setup, QA, features, backtest, diagnostics).
- `data/raw`: fichiers CSV source (intraday + daily).
- `data/processed`: données nettoyées / enrichies.
- `data/exports`: exports et visuels.
- `src/`: logique métier (config, data, features, strategy, engine, analytics, visualization, utils).
- `tests/`: tests unitaires de base.

## Hypothèses de marché codées

- Timezone par défaut: `America/New_York`.
- Tick size NQ: `0.25`.
- Tick value paramétrable (défaut `5 USD`).
- Les timestamps représentent le début de période.
- Les minutes manquantes en intraday ne sont pas automatiquement traitées comme une erreur (barres volume nul absentes).

## Installation

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer le notebook central

```bash
jupyter notebook notebooks/research_main.ipynb
```

## Exécuter les tests

```bash
PYTHONPATH=. pytest -q
```

## Extensions recommandées

1. Ajouter gestion multi-instrument et metadata contractuelles.
2. Ajouter modèles de slippage avancés (volatility/liquidity-aware).
3. Ajouter walk-forward et robustesse out-of-sample.
4. Ajouter support exécution live (broker adapters + risk manager).
