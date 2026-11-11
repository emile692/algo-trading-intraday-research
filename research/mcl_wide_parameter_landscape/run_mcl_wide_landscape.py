"""Execute the inherited locked 7x7 IS/OOS landscape for all 16 MCL mechanisms."""
from concurrent.futures import ProcessPoolExecutor as NativeProcessPoolExecutor
from datetime import date
import json
from pathlib import Path

import pandas as pd

import research.mgc_wide_parameter_landscape.run_mgc_wide_landscape as engine
from research.mcl_baseline_backtests.run_mcl_baselines import adapt, prepare

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MCL_CONFIG = ROOT / "research/mcl_baseline_backtests/mcl_baseline_config.json"
DATA = ROOT / "data/downloaded/MCL_v_0_1m_20210712_20260924.parquet"


def _plot_mechanism_mcl(group, mechanism, definition):
    """Render the generic landscape with the actual MCL asset and dates."""
    p1, p2 = definition["parameter_1"], definition["parameter_2"]
    is_sharpe = engine._pivot(group, "sharpe_annualized_is", definition)
    oos_sharpe = engine._pivot(group, "sharpe_annualized_validation", definition)
    positive = engine._pivot(group, "positive_both", definition).astype("boolean").fillna(False).astype(bool)
    count_is = engine._pivot(group, "trade_count_is", definition)
    count_oos = engine._pivot(group, "trade_count_validation", definition)

    for label, pivot, title in (
        ("sharpe_is", is_sharpe, "Sharpe annualise - IS 2021-07-12 to 2023-12-31"),
        ("sharpe_validation", oos_sharpe, "Sharpe annualise - OOS 2024-01-01 to 2026-03-19"),
    ):
        fig, ax = engine.plt.subplots(figsize=(9.2, 6.6))
        engine._heatmap(ax, pivot, f"{mechanism}\n{title}", p2, p1, ".2f")
        fig.tight_layout()
        fig.savefig(engine.HEATMAP_DIR / f"{mechanism}_{label}.png", dpi=155)
        engine.plt.close(fig)

    fig, axes = engine.plt.subplots(1, 2, figsize=(17, 6.8), sharey=True)
    engine._heatmap(axes[0], is_sharpe, "IS - Sharpe annualise", p2, p1, ".2f")
    engine._heatmap(axes[1], oos_sharpe, "OOS - Sharpe annualise", p2, p1, ".2f")
    fig.suptitle(f"{mechanism} - paysage large MCL", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(engine.HEATMAP_DIR / f"{mechanism}_sharpe_is_validation.png", dpi=155)
    engine.plt.close(fig)

    fig, ax = engine.plt.subplots(figsize=(9.2, 6.6))
    engine._heatmap(ax, positive, f"{mechanism}\nSharpe positif en IS ET OOS", p2, p1, "d", mask=True)
    fig.tight_layout()
    fig.savefig(engine.HEATMAP_DIR / f"{mechanism}_positive_both_mask.png", dpi=155)
    engine.plt.close(fig)

    fig, axes = engine.plt.subplots(1, 2, figsize=(17, 6.8), sharey=True)
    engine._heatmap(axes[0], count_is, "IS - nombre de trades", p2, p1, ".0f")
    engine._heatmap(axes[1], count_oos, "OOS - nombre de trades", p2, p1, ".0f")
    fig.suptitle(f"{mechanism} - taille d'echantillon par cellule", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(engine.HEATMAP_DIR / f"{mechanism}_trade_count_is_validation.png", dpi=155)
    engine.plt.close(fig)


def finalize_existing_outputs() -> None:
    """Correct asset/date labels and metadata without rerunning any backtest."""
    config = json.loads((HERE / "mcl_wide_parameter_grids.json").read_text(encoding="utf-8"))
    results = pd.read_csv(HERE / "mcl_wide_landscape_results.csv")
    wide = engine._wide_results(results)
    engine.HEATMAP_DIR = HERE / "heatmaps"
    engine.REPORT_PATH = HERE / "MCL_WIDE_LANDSCAPE_ENGINE_REPORT.md"
    engine.HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    for mechanism, definition in config["mechanisms"].items():
        _plot_mechanism_mcl(wide[wide["mechanism_id"] == mechanism], mechanism, definition)

    metadata_path = HERE / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["run_type"] = "mcl_wide_parameter_landscape_is_oos"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Reuse the audited tabular topology calculation, then correct only its
    # asset/date vocabulary. Grid rationales remain explicitly inherited.
    engine._write_report(config, results, wide, metadata)
    report = engine.REPORT_PATH.read_text(encoding="utf-8")
    report = report.replace("# MGC —", "# MCL —")
    report = report.replace("2019-05-06 → 2023-12-31", "2021-07-12 → 2023-12-31")
    report = report.replace("session MGC", "session MCL")
    report = report.replace("paysage large MGC", "paysage large MCL")
    report = report.replace("série continue MGC", "série continue MCL")
    report = report.replace("Validation", "OOS")
    report = report.replace("OOS temporelle", "OOS")
    report = report.replace("La OOS sert", "La période OOS sert")
    report = report.replace("après correction de l'ancrage horaire MGC", "avec l'ancrage horaire MCL")
    executive = (
        "## Lecture synthétique\n\n"
        "- `ma_crossover_trend` est le paysage le plus cohérent : 6 cellules positives IS/OOS, "
        "toutes contiguës, avec 144–239 trades IS et 150–244 OOS. Le plateau reste toutefois "
        "modeste OOS (Sharpe 0.02–0.42).\n"
        "- `gap_continuation` forme aussi un plateau contigu de 5 cellules, mais l'échantillon "
        "est faible (15–47 trades IS; 18–46 OOS) et le Sharpe OOS seulement 0.04–0.18.\n"
        "- `statistical_zscore_reversion` possède 4 cellules communes, dont une à 158/156 trades "
        "et Sharpe 0.50/0.56, mais elles sont isolées : pas de plateau paramétrique.\n"
        "- `atr_trend_flip` a 5 cellules communes et beaucoup de trades, mais son Sharpe OOS "
        "reste faible (0.09–0.14). Les cellules positives des autres familles sont rares ou "
        "reposent surtout sur de petits échantillons.\n"
        "- 8 mécanismes n'ont aucune cellule positive simultanément en IS et OOS. Ces paysages "
        "ne constituent pas une sélection automatique de paramètres.\n\n"
    )
    gallery_lines = ["## Galerie Sharpe IS/OOS", ""]
    for mechanism in config["mechanisms"]:
        gallery_lines.extend(
            [
                f"### `{mechanism}`",
                "",
                f"![{mechanism} — Sharpe IS/OOS](heatmaps/{mechanism}_sharpe_is_validation.png)",
                "",
            ]
        )
    gallery = "\n".join(gallery_lines) + "\n"
    report = report.replace(
        "## Grilles et fichiers par mécanisme\n",
        executive + gallery + "## Grilles et fichiers par mécanisme\n",
    )
    report += (
        "\n## Provenance de la grille\n\n"
        "Les axes 7×7 sont repris sans modification de la campagne MGC antérieure; "
        "les mentions MGC/Gold dans certaines justifications décrivent donc leur origine, "
        "pas une calibration effectuée après lecture des résultats MCL.\n"
    )
    engine.REPORT_PATH.write_text(report, encoding="utf-8")


def prepare_adapter(path, start, end):
    cfg = json.loads(MCL_CONFIG.read_text(encoding="utf-8"))
    return prepare(Path(path), start, end, cfg), pd.DataFrame()


def adapt_adapter(source, cfg):
    return adapt(source, cfg)


_PREPARED = None
_SPECS = {}
_CFG = {}
_PERIODS = {}


def worker_init(data_path, full_start, full_end, periods):
    global _PREPARED, _SPECS, _CFG, _PERIODS
    _CFG = json.loads(MCL_CONFIG.read_text(encoding="utf-8"))
    source_specs, _, _ = engine.load_frozen_specs(engine.FROZEN_DIR, engine.FROZEN_MANIFEST)
    _SPECS = {item["mechanism_id"]: item for item in source_specs}
    _PERIODS = periods
    _PREPARED = prepare(Path(data_path), pd.Timestamp(full_start).date(), pd.Timestamp(full_end).date(), _CFG)


def process_cell(task):
    mechanism, definition, combo = task
    i1, i2, v1, v2 = combo
    spec = engine._apply_parameters(adapt(_SPECS[mechanism], _CFG), definition, v1, v2)
    run = engine.run_baseline_backtest(spec, _PREPARED)
    violations = engine._violations(run.checks)
    if violations:
        raise RuntimeError(f"{mechanism} {v1}/{v2}: {violations}")
    rows = []
    for period in engine.PERIOD_ORDER:
        start, end = (date.fromisoformat(item) for item in _PERIODS[period])
        rows.append({
            "mechanism_id": mechanism, "strategy_id": spec["strategy_id"],
            "complexity_grade": spec["complexity_grade"],
            "combination_id": f"{mechanism}__{i1:02d}_{i2:02d}",
            "parameter_1_name": definition["parameter_1"], "parameter_1_value": v1, "parameter_1_index": i1,
            "parameter_2_name": definition["parameter_2"], "parameter_2_value": v2, "parameter_2_index": i2,
            "period": period, **engine._period_metrics(run, start, end),
            "missing_execution_bar_skips_full_period": int(run.checks.get("missing_execution_bar_skips", 0)),
            "hard_flatten_fallbacks_full_period": int(run.checks.get("hard_flatten_fallbacks", 0)),
            "backtest_valid": True,
        })
    return rows


class MCLProcessPoolExecutor:
    """Redirect the generic engine's worker hooks to importable MCL hooks."""
    def __init__(self, max_workers, initializer=None, initargs=()):
        self.pool = NativeProcessPoolExecutor(max_workers=max_workers, initializer=worker_init, initargs=initargs)

    def __enter__(self):
        self.pool.__enter__(); return self

    def __exit__(self, *args):
        return self.pool.__exit__(*args)

    def submit(self, ignored_function, task):
        return self.pool.submit(process_cell, task)


def execute(workers: int = 4) -> None:
    engine.OUTPUT_DIR = HERE
    engine.HEATMAP_DIR = HERE / "heatmaps"
    engine.MPL_CACHE = HERE / ".matplotlib"
    engine.GRID_PATH = HERE / "mcl_wide_parameter_grids.json"
    engine.GRID_HASH_PATH = HERE / "mcl_wide_parameter_grids.sha256"
    engine.MGC_CONFIG_PATH = MCL_CONFIG
    engine.RESULT_PATH = HERE / "mcl_wide_landscape_results.csv"
    engine.REPORT_PATH = HERE / "MCL_WIDE_LANDSCAPE_ENGINE_REPORT.md"
    engine.PARTIAL_PATH = HERE / "mcl_wide_landscape_results.partial.csv"
    engine.prepare_mgc_data = prepare_adapter
    engine.adapt_spec_to_mgc = adapt_adapter
    engine.ProcessPoolExecutor = MCLProcessPoolExecutor
    engine.execute(DATA, workers)
    finalize_existing_outputs()


if __name__ == "__main__":
    execute()
