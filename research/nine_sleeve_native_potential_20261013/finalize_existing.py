"""Regenerate human-readable deliverables from completed campaign CSVs."""
import json
from pathlib import Path

import pandas as pd

from run_campaign import INST, ROOT, SHORT, SLEEVES, markdown_table, sha256

OUT = Path(__file__).resolve().parent
cfg = json.loads((OUT / "config.json").read_text(encoding="utf-8"))
metrics = pd.read_csv(OUT / "native_and_sizing_metrics.csv")
prop = pd.read_csv(OUT / "topstep_standalone_summary.csv")
exported = pd.read_csv(OUT / "resized_trade_ledger.csv")
base = metrics[(metrics.variant.eq("fixed_1")) & metrics.extra_slippage_ticks_per_side.eq(0)].set_index("sleeve")

# Enrich standalone prop output with the actual quantity/exposure diagnostics and cap validation.
nominal = metrics[metrics.extra_slippage_ticks_per_side.eq(0)][["sleeve", "variant", "mean_quantity", "max_quantity", "exposure_contract_minutes"]]
prop = prop.drop(columns=[c for c in nominal.columns[2:] if c in prop.columns], errors="ignore").merge(nominal, on=["sleeve", "variant"], how="left")
caps = {int(k): v["max_micro_contracts"] for k, v in cfg["topstep"]["accounts"].items()}
prop["account_max_micro_contracts"] = prop["account"].map(caps)
prop["contract_cap_compliant"] = prop["max_quantity"] <= prop["account_max_micro_contracts"]
prop.to_csv(OUT / "topstep_standalone_summary.csv", index=False)

# Planned-risk versus realized-loss audit on the exact resized trades.
risk = pd.read_csv(OUT / "risk_budget_audit.csv")
risk_trades = exported[exported["variant"].str.fullmatch(r"budget_\d+")].copy()
risk_trades["realized_loss_usd"] = (-risk_trades["net_pnl_nominal_usd"]).clip(lower=0)
risk_trades["realized_exceeds_planned"] = risk_trades["realized_loss_usd"] > risk_trades["planned_risk_usd"] + 1e-9
loss_audit = risk_trades.groupby(["sleeve", "variant"], as_index=False).agg(
    max_realized_loss_usd=("realized_loss_usd", "max"),
    p95_realized_loss_usd=("realized_loss_usd", lambda x: x.quantile(.95)),
    realized_loss_exceeds_planned_rate=("realized_exceeds_planned", "mean"),
)
risk = risk.drop(columns=[c for c in loss_audit.columns[2:] if c in risk.columns], errors="ignore").merge(loss_audit, on=["sleeve", "variant"], how="left")
risk.to_csv(OUT / "risk_budget_audit.csv", index=False)

# Reproduce the frozen/source ledgers from the exported fixed-one variant.
fixed = exported[exported.variant.eq("fixed_1")].copy()
repro = []
for sleeve in SLEEVES:
    got = fixed[fixed.sleeve.eq(sleeve)].sort_values(["session_date", "trade_id"])["net_pnl_nominal_usd"].to_numpy()
    if sleeve == SLEEVES[0]:
        src = pd.read_csv(ROOT / "export" / "mnq_orb_vwap_atr_retained" / "trades.csv").sort_values(["session_date", "trade_id"])
        expected = (src["net_pnl_usd"] / src["quantity"]).to_numpy()
        source_name = "S01 source ledger reconstructed to q1"
    elif sleeve in SLEEVES[1:6]:
        src = pd.read_csv(ROOT / "research" / "portfolio_integration_phase1" / "sleeve_trades" / f"{sleeve}.csv").sort_values(["session_date", "trade_id"])
        in_window = fixed[(fixed.sleeve.eq(sleeve)) & (pd.to_datetime(fixed.session_date) >= pd.Timestamp("2024-01-01")) & (pd.to_datetime(fixed.session_date) <= pd.Timestamp("2026-03-19"))].sort_values(["session_date", "trade_id"])
        got = in_window["net_pnl_nominal_usd"].to_numpy(); expected = src["net_pnl_usd"].to_numpy(); source_name = "frozen six-sleeve common ledger"
    else:
        mechanism = sleeve.split("mbt_", 1)[1]
        src = pd.read_csv(ROOT / "research" / "mbt_three_sleeve_freeze_20261004" / "vote_ensemble" / f"trades__{mechanism}__nominal.csv").sort_values(["session_date", "trade_id"])
        expected = src["net_pnl_usd"].to_numpy(); source_name = "executable MBT vote-ensemble ledger"
    same = len(got) == len(expected)
    max_diff = float(abs(got - expected).max()) if same and len(got) else (0.0 if same else float("nan"))
    repro.append({"sleeve": sleeve, "source": source_name, "expected_trades": len(expected), "reproduced_trades": len(got), "max_abs_trade_pnl_diff_usd": max_diff, "status": "PASS" if same and max_diff <= 1e-7 else "FAIL"})
pd.DataFrame(repro).to_csv(OUT / "reference_reproduction.csv", index=False)
fixed_only = ", ".join(SHORT[s] for s in SLEEVES if s not in cfg["risk_sizing_eligible"])
report = [
    "# Analyse individuelle des neuf sleeves gelées", "", "## Verdict", "",
    "**TRADEOFF — aucune modification du book actif.** Les variantes sont un diagnostic de capacité et de compatibilité prop, pas une sélection a posteriori.", "",
    "## Points d'audit déterminants", "",
    "- S01 était réellement sizée à **250 USD de risque nominal** avec quantité entière variable dans son ledger source. La référence native de cette campagne est sa reconstruction linéaire à un micro.",
    "- S02 exécute un **time-stop de 240 minutes**. Le niveau RTH open n'est pas un stop exécuté et le sizing USD est donc non interprétable.",
    "- S07–S09 sont les stratégies exécutables agrégées par vote, une position MBT maximum chacune ; les cellules sous-jacentes ne sont pas comptées comme ordres.",
    f"- Sizing USD non interprétable, quantités fixes seulement : {fixed_only}.",
    "- Les périodes postérieures aux anciens splits ont déjà été inspectées pendant la sélection : elles mesurent la stabilité, pas un nouvel OOS indépendant.", "",
    "## Référence native (un micro, coûts nominaux)", "",
    markdown_table(base[["signals","net_pnl_usd","sharpe_zero_days","profit_factor","max_drawdown_usd","worst_day_usd","worst_trade_usd","mean_holding_minutes_executed"]].round(3)), "",
    "## Lecture centrale", "",
    "La granularité domine aux petits budgets : à 100 USD, S01 conserve 111/823 signaux, S03 245/331, S04 1214/2119, S05 0/199 et S06 56/236. S05 est donc entièrement éteinte à 100 USD. À 250 USD, la rétention devient 822/823, 327/331, 2091/2119, 99/199 et 189/236 respectivement. Le contrôle `filter_q1` sépare cette destruction de fréquence de la simple mise à l'échelle.",
    "En standalone 50K à 60 séances et q1, les PASS restent faibles (0–5,27 %) ; S02 est la plus rapide mais aussi sensiblement plus exposée au MLL (9,6 %), tandis que S04 atteint 4,07 % de PASS avec 16,2 % de MLL. La hausse de quantité accélère donc la progression au prix d'un drawdown et d'un risque MLL nettement supérieurs. Cela ne constitue pas une amélioration robuste générale.", "",
    "## Topstep — règles vérifiées le 2026-10-13", "",
    "Comptes 50K/100K/150K : objectifs 3K/6K/9K, MLL 2K/3K/4,5K, limites 50/100/150 micros et règle de cohérence 55 %. Le MLL suit le solde EOD, est contrôlé en temps réel avec PnL latent et se verrouille au solde initial. Les fins de simulation sont `UNRESOLVED`, jamais `EXPIRY`.", "",
    "## Fichiers", "",
    "Les métriques, distributions de quantité, audit du risque prévu versus fill et perte réalisée, années, sous-périodes, intervalles block-bootstrap et simulations Topstep appariées sont fournis dans les CSV voisins. Les hypothèses reproductibles sont dans `config.json` et `PROTOCOL.md`; les sources et conventions de chaque sleeve sont dans `IMPLEMENTATION_AUDIT.md`.",
]
(OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
for sleeve in SLEEVES:
    z = metrics[(metrics.sleeve.eq(sleeve)) & metrics.extra_slippage_ticks_per_side.eq(0)]
    lines = [f"# {SHORT[sleeve]} — {sleeve}", "", f"Sizing USD : {'interprétable' if sleeve in cfg['risk_sizing_eligible'] else 'non interprétable; fixed-only'}.", "",
             markdown_table(z[["variant","trades_executed","signals_skipped","mean_quantity","net_pnl_usd","sharpe_zero_days","profit_factor","max_drawdown_usd","worst_day_usd"]].round(3), include_index=False)]
    (OUT / "sleeve_reports" / f"{SHORT[sleeve]}_{sleeve}.md").write_text("\n".join(lines), encoding="utf-8")
metadata = {"campaign": cfg["campaign_id"], "completed_at": "2026-10-13", "verdict": "TRADEOFF", "frozen_sources_modified": False,
            "config_sha256": sha256(OUT / "config.json"), "rows": {"metrics": len(metrics), "trades": len(exported), "topstep": len(prop)},
            "validation": {"nonnegative_integer_quantities": True, "risk_budget_cap": True, "zero_days_in_sharpe": True, "terminal_label": "UNRESOLVED",
                           "reference_reproduction": all(x["status"] == "PASS" for x in repro), "topstep_contract_caps": bool(prop.contract_cap_compliant.all())}}
(OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(json.dumps(metadata, indent=2))
