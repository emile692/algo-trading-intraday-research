"""Compare Manuel's four daily-PnL sleeves with the active six-sleeve book."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


MANUEL = ["US30", "GER40", "US100", "AUS200"]
EMILE_LONG = [
    "mnq_orb_vwap_atr_pnl_usd",
    "mnq_volatility_breakout_pnl_usd",
    "mnq_compression_breakout_pnl_usd",
    "mnq_atr_trend_flip_pnl_usd",
    "mgc_volatility_breakout_pnl_usd",
    "mgc_gap_continuation_pnl_usd",
]
EMILE = ["MNQ_ORB", "MNQ_VOL", "MNQ_COMP", "MNQ_TREND", "MGC_VOL", "MGC_GAP"]
ANNUAL = 252


def load_inputs(manuel_path: Path, emile_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    manuel = pd.read_csv(manuel_path)
    manuel["date"] = pd.to_datetime(manuel.pop("Date"), format="%m/%d/%Y")
    manuel = manuel.set_index("date").sort_index()
    emile = pd.read_csv(emile_path)
    emile["date"] = pd.to_datetime(emile.pop("session_date"))
    emile = emile.set_index("date").sort_index().rename(columns=dict(zip(EMILE_LONG, EMILE)))
    return manuel, emile


def business_grid(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="B", name="date")
    return frame.reindex(idx).fillna(0.0)


def drawdown(series: pd.Series) -> pd.Series:
    equity = series.cumsum()
    return equity - equity.cummax().clip(lower=0.0)


def metrics(series: pd.Series) -> dict[str, float]:
    x = series.astype(float)
    active = x.ne(0)
    std = x.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(x.to_numpy(), 0.0) ** 2))
    gross_profit = x[x > 0].sum()
    gross_loss = -x[x < 0].sum()
    ann_pnl = x.mean() * ANNUAL
    max_dd = -drawdown(x).min()
    return {
        "start": x.index.min().date().isoformat(),
        "end": x.index.max().date().isoformat(),
        "days": len(x),
        "active_days": int(active.sum()),
        "active_pct": 100.0 * active.mean(),
        "total_pnl": x.sum(),
        "annualized_pnl": ann_pnl,
        "annualized_vol": std * np.sqrt(ANNUAL),
        "sharpe_0rf": x.mean() / std * np.sqrt(ANNUAL) if std else np.nan,
        "sortino_0target": x.mean() / downside * np.sqrt(ANNUAL) if downside else np.nan,
        "max_drawdown": max_dd,
        "pnl_to_maxdd": ann_pnl / max_dd if max_dd else np.nan,
        "active_hit_rate_pct": 100.0 * (x[active] > 0).mean() if active.any() else np.nan,
        "profit_factor": gross_profit / gross_loss if gross_loss else np.nan,
        "best_day": x.max(),
        "worst_day": x.min(),
        "skew": x.skew(),
    }


def risk_normalize(frame: pd.DataFrame) -> pd.DataFrame:
    vol = frame.std(ddof=1).replace(0, np.nan)
    return frame.divide(vol)


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3 or x.std(ddof=1) == 0 or y.std(ddof=1) == 0:
        return np.nan
    return x.corr(y)


def pair_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in MANUEL:
        for e in EMILE:
            both = data[m].ne(0) & data[e].ne(0)
            rows.append(
                {
                    "manuel_sleeve": m,
                    "emile_sleeve": e,
                    "pearson_all_days": safe_corr(data[m], data[e]),
                    "spearman_all_days": data[m].corr(data[e], method="spearman"),
                    "both_active_days": int(both.sum()),
                    "pearson_both_active": safe_corr(data.loc[both, m], data.loc[both, e]),
                    "same_sign_when_both_active_pct": (
                        100.0 * (np.sign(data.loc[both, m]) == np.sign(data.loc[both, e])).mean()
                        if both.any()
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def regression_uniqueness(data_z: pd.DataFrame) -> pd.DataFrame:
    x = data_z[EMILE].to_numpy()
    split = len(data_z) // 2
    rows = []
    for name in MANUEL:
        y = data_z[name].to_numpy()
        x1 = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(x1, y, rcond=None)[0]
        fitted = x1 @ beta
        sst = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - np.sum((y - fitted) ** 2) / sst
        beta_train = np.linalg.lstsq(x1[:split], y[:split], rcond=None)[0]
        pred = x1[split:] @ beta_train
        y_test = y[split:]
        test_sst = np.sum((y_test - y_test.mean()) ** 2)
        oos_r2 = 1.0 - np.sum((y_test - pred) ** 2) / test_sst
        rows.append(
            {
                "manuel_sleeve": name,
                "in_sample_r2_vs_six": r2,
                "second_half_oos_r2": oos_r2,
                "unexplained_variance_pct": 100.0 * (1.0 - r2),
            }
        )
    return pd.DataFrame(rows)


def blend_table(emile_book: pd.Series, manuel_book: pd.Series) -> pd.DataFrame:
    e = emile_book / emile_book.std(ddof=1)
    m = manuel_book / manuel_book.std(ddof=1)
    rows = []
    for w in [0.0, 0.10, 0.20, 0.25, 1 / 3, 0.50, 1.0]:
        p = (1.0 - w) * e + w * m
        met = metrics(p)
        rows.append(
            {
                "manuel_risk_budget_pct": 100.0 * w,
                "sharpe_0rf": met["sharpe_0rf"],
                "annualized_pnl_risk_units": met["annualized_pnl"],
                "annualized_vol_risk_units": met["annualized_vol"],
                "max_drawdown_risk_units": met["max_drawdown"],
                "pnl_to_maxdd": met["pnl_to_maxdd"],
            }
        )
    return pd.DataFrame(rows)


def bootstrap_books(emile_book: pd.Series, manuel_book: pd.Series, seed: int = 1701) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    e = (emile_book / emile_book.std(ddof=1)).to_numpy()
    m = (manuel_book / manuel_book.std(ddof=1)).to_numpy()
    n, block, reps = len(e), 20, 2000
    starts = np.arange(n - block + 1)
    correlations = np.empty(reps)
    sharpe_deltas = np.empty(reps)
    for i in range(reps):
        chosen = rng.choice(starts, size=int(np.ceil(n / block)), replace=True)
        idx = np.concatenate([np.arange(s, s + block) for s in chosen])[:n]
        eb, mb = e[idx], m[idx]
        correlations[i] = np.corrcoef(eb, mb)[0, 1]
        base_sharpe = eb.mean() / eb.std(ddof=1) * np.sqrt(ANNUAL)
        mix = 0.8 * eb + 0.2 * mb
        sharpe_deltas[i] = mix.mean() / mix.std(ddof=1) * np.sqrt(ANNUAL) - base_sharpe
    rows = []
    for metric_name, values in [
        ("book_correlation", correlations),
        ("sharpe_delta_80_20_vs_emile", sharpe_deltas),
    ]:
        q = np.quantile(values, [0.025, 0.5, 0.975])
        rows.append(
            {
                "metric": metric_name,
                "estimate": np.corrcoef(e, m)[0, 1]
                if metric_name == "book_correlation"
                else blend_table(emile_book, manuel_book).iloc[2].sharpe_0rf
                - blend_table(emile_book, manuel_book).iloc[0].sharpe_0rf,
                "ci_2_5": q[0],
                "bootstrap_median": q[1],
                "ci_97_5": q[2],
                "block_days": block,
                "replications": reps,
            }
        )
    return pd.DataFrame(rows)


def regime_table(data_z: pd.DataFrame, e_book: pd.Series, m_book: pd.Series) -> pd.DataFrame:
    eq10 = e_book.quantile(0.10)
    mq10 = m_book.quantile(0.10)
    e_stress = e_book <= eq10
    m_stress = m_book <= mq10
    rows = []
    for m in MANUEL:
        x = data_z.loc[e_stress, m]
        rows.append(
            {
                "test": "Manuel sleeve during worst 10% Emile-book days",
                "series": m,
                "observations": len(x),
                "mean_daily_risk_units": x.mean(),
                "positive_days_pct": 100.0 * (x > 0).mean(),
            }
        )
    x = e_book.loc[m_stress]
    rows.append(
        {
            "test": "Emile book during worst 10% Manuel-book days",
            "series": "EMILE_BOOK",
            "observations": len(x),
            "mean_daily_risk_units": x.mean(),
            "positive_days_pct": 100.0 * (x > 0).mean(),
        }
    )
    return pd.DataFrame(rows)


def lead_lag(e_book: pd.Series, m_book: pd.Series) -> pd.DataFrame:
    rows = []
    # Positive lag means Manuel is shifted later relative to Emile.
    for lag in range(-3, 4):
        rows.append({"manuel_shift_business_days": lag, "correlation": e_book.corr(m_book.shift(lag))})
    return pd.DataFrame(rows)


def yearly_diagnostics(data: pd.DataFrame, data_z: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, block in data.groupby(data.index.year):
        for name in MANUEL:
            rows.append(
                {
                    "year": int(year),
                    "series": name,
                    "pnl": block[name].sum(),
                    "active_days": int(block[name].ne(0).sum()),
                    "sharpe_0rf": metrics(block[name])["sharpe_0rf"],
                }
            )
        zblock = data_z.loc[block.index]
        e = zblock[EMILE].mean(axis=1)
        m = zblock[MANUEL].mean(axis=1)
        rows.append(
            {
                "year": int(year),
                "series": "BOOK_CORRELATION",
                "pnl": np.nan,
                "active_days": int(m.ne(0).sum()),
                "sharpe_0rf": e.corr(m),
            }
        )
    return pd.DataFrame(rows)


def concentration_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in MANUEL:
        x = data[name]
        active = x[x.ne(0)]
        positive_total = x[x > 0].sum()
        rows.append(
            {
                "series": name,
                "first_active_date": active.index.min().date().isoformat(),
                "last_active_date": active.index.max().date().isoformat(),
                "top_5_days_pct_total_pnl": 100.0 * x.nlargest(5).sum() / x.sum(),
                "top_10_days_pct_gross_profit": 100.0 * x.nlargest(10).sum() / positive_total,
                "lag1_autocorrelation": x.autocorr(1),
            }
        )
    return pd.DataFrame(rows)


def save_figures(
    out: Path,
    manuel_full: pd.DataFrame,
    data_z: pd.DataFrame,
    e_book: pd.Series,
    m_book: pd.Series,
    corr: pd.DataFrame,
    blends: pd.DataFrame,
    yearly: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    manuel_full[MANUEL].cumsum().plot(ax=axes[0], linewidth=1.4)
    axes[0].set_title("Manuel — courbes de PnL cumulées (unités brutes fournies)")
    axes[0].set_ylabel("PnL cumulé")
    drawdowns = manuel_full[MANUEL].apply(drawdown)
    drawdowns.plot(ax=axes[1], linewidth=1.1)
    axes[1].set_title("Manuel — drawdowns")
    axes[1].set_ylabel("Drawdown")
    fig.tight_layout()
    fig.savefig(out / "01_manuel_equity_drawdown.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 9))
    dist = (1.0 - corr.clip(-1, 1)) / 2.0
    np.fill_diagonal(dist.values, 0.0)
    order = leaves_list(linkage(squareform(dist.values, checks=False), method="average"))
    ordered = corr.iloc[order, order]
    sns.heatmap(ordered, cmap="vlag", center=0, vmin=-0.5, vmax=0.5, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Corrélations quotidiennes et clustering — fenêtre commune")
    fig.tight_layout()
    fig.savefig(out / "02_correlation_cluster.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5))
    rolling = e_book.rolling(63, min_periods=40).corr(m_book)
    ax.plot(rolling.index, rolling, color="#6a51a3", linewidth=1.3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(e_book.corr(m_book), color="#de2d26", linestyle="--", label="corrélation globale")
    ax.set_ylim(-1, 1)
    ax.set_title("Corrélation roulante 63 jours — livres equal-risk")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "03_rolling_book_correlation.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    pd.DataFrame({"Emile_6": e_book, "Manuel_4": m_book}).cumsum().plot(ax=axes[0])
    axes[0].set_title("Livres equal-risk — PnL cumulé en unités de risque")
    pd.DataFrame({"Emile_6": drawdown(e_book), "Manuel_4": drawdown(m_book)}).plot(ax=axes[1])
    axes[1].set_title("Livres equal-risk — drawdowns en unités de risque")
    fig.tight_layout()
    fig.savefig(out / "04_books_equity_drawdown.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    plot = blends[blends.manuel_risk_budget_pct <= 50]
    ax.plot(plot.manuel_risk_budget_pct, plot.sharpe_0rf, marker="o")
    for _, row in plot.iterrows():
        ax.annotate(f"{row.sharpe_0rf:.2f}", (row.manuel_risk_budget_pct, row.sharpe_0rf), xytext=(0, 7), textcoords="offset points", ha="center")
    ax.set_xlabel("Budget de risque Manuel (%)")
    ax.set_ylabel("Sharpe (rf=0)")
    ax.set_title("Impact marginal d'une allocation au livre Manuel")
    fig.tight_layout()
    fig.savefig(out / "05_blend_sharpe.png", dpi=170)
    plt.close(fig)

    annual = yearly[yearly.series.isin(MANUEL)].pivot(index="year", columns="series", values="pnl")
    activity = yearly[yearly.series.isin(MANUEL)].pivot(index="year", columns="series", values="active_days")
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    annual.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Manuel — PnL annuel sur la fenêtre commune")
    axes[0].set_ylabel("PnL brut")
    activity.plot(kind="bar", ax=axes[1])
    axes[1].set_title("Manuel — nombre de jours actifs par année")
    axes[1].set_ylabel("Jours actifs")
    axes[1].set_xlabel("Année")
    fig.tight_layout()
    fig.savefig(out / "06_yearly_stability.png", dpi=170)
    plt.close(fig)


def fmt_table(df: pd.DataFrame, columns: list[str], decimals: int = 2) -> str:
    view = df[columns].copy()
    for column in view.select_dtypes(include=[np.number]).columns:
        view[column] = view[column].map(lambda value: f"{value:.{decimals}f}" if pd.notna(value) else "")
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = [
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        for row in view.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def build_report(
    out: Path,
    manuel_raw: pd.DataFrame,
    full_metrics: pd.DataFrame,
    overlap_metrics: pd.DataFrame,
    pairwise: pd.DataFrame,
    uniqueness: pd.DataFrame,
    blends: pd.DataFrame,
    bootstrap: pd.DataFrame,
    regimes: pd.DataFrame,
    leadlags: pd.DataFrame,
    yearly: pd.DataFrame,
    concentration: pd.DataFrame,
    post_metrics: pd.DataFrame,
    post_book_metrics: dict[str, float] | None,
    e_book: pd.Series,
    m_book: pd.Series,
) -> None:
    cross = pairwise.copy()
    mean_abs_by_m = cross.groupby("manuel_sleeve").pearson_all_days.apply(lambda s: s.abs().mean())
    max_abs_by_m = cross.assign(a=cross.pearson_all_days.abs()).groupby("manuel_sleeve").a.max()
    book_corr = e_book.corr(m_book)
    rolling = e_book.rolling(63, min_periods=40).corr(m_book).dropna()
    best_lag = leadlags.loc[leadlags.correlation.abs().idxmax()]
    b20 = blends.loc[np.isclose(blends.manuel_risk_budget_pct, 20)].iloc[0]
    b0 = blends.iloc[0]
    boot_corr = bootstrap.loc[bootstrap.metric == "book_correlation"].iloc[0]
    boot_delta = bootstrap.loc[bootstrap.metric == "sharpe_delta_80_20_vs_emile"].iloc[0]
    weekend_rows = int((manuel_raw.index.dayofweek >= 5).sum())
    weekend_nonzero = int(manuel_raw.loc[manuel_raw.index.dayofweek >= 5, MANUEL].ne(0).any(axis=1).sum())
    total_error = (manuel_raw["TOTAL"] - manuel_raw[MANUEL].sum(axis=1)).abs().max()
    annual = yearly[yearly.series.isin(MANUEL)].pivot(index="year", columns="series", values="pnl").reset_index()
    activity = yearly[yearly.series.isin(MANUEL)].pivot(index="year", columns="series", values="active_days").reset_index()
    yearly_book_corr = yearly[yearly.series == "BOOK_CORRELATION"].sharpe_0rf

    if abs(book_corr) < 0.10 and uniqueness.in_sample_r2_vs_six.max() < 0.10:
        verdict = "forte orthogonalité statistique ; moteur causal distinct plausible mais non démontré"
    elif abs(book_corr) < 0.25 and uniqueness.in_sample_r2_vs_six.max() < 0.25:
        verdict = "orthogonalité utile, mais moteur distinct encore à confirmer"
    else:
        verdict = "orthogonalité insuffisante pour conclure à un moteur distinct"

    text = f"""# Analyse des 4 sleeves de Manuel vs les 6 sleeves actives

## Verdict exécutif

**Conclusion : {verdict}.** Sur la fenêtre commune, la corrélation des deux livres equal-risk est **{book_corr:.3f}** (IC bootstrap par blocs 95 % : **[{boot_corr.ci_2_5:.3f}, {boot_corr.ci_97_5:.3f}]**). Les régressions des sleeves Manuel sur les six sleeves actuelles n'expliquent que **{100*uniqueness.in_sample_r2_vs_six.min():.1f} % à {100*uniqueness.in_sample_r2_vs_six.max():.1f} %** de leur variance in-sample.

Une allocation de **20 % du budget de risque** au livre Manuel fait passer le Sharpe indicatif de **{b0.sharpe_0rf:.2f} à {b20.sharpe_0rf:.2f}** (delta **{b20.sharpe_0rf-b0.sharpe_0rf:+.2f}** ; IC bootstrap 95 % **[{boot_delta.ci_2_5:+.2f}, {boot_delta.ci_97_5:+.2f}]**). Cette estimation est historique, avant coûts et non OOS.

Le point de prudence principal : les quatre séries de Manuel sont des PnL agrégés, sans notionnel, coûts, heures de prise de signal ni ledger de trades. L'orthogonalité de PnL est observable ; l'affirmation d'un *moteur causal* différent exige encore le protocole d'échange listé en fin de note.

**Alerte récente :** après la fin de la fenêtre comparable, `AUS200` subit un PnL de **{post_metrics.set_index('series').loc['AUS200', 'total_pnl']:.0f}** et un drawdown de **{post_metrics.set_index('series').loc['AUS200', 'max_drawdown']:.0f}**. Le livre Manuel equal-risk, calibré avec les volatilités de la fenêtre commune, affiche sur ce pseudo-holdout un Sharpe de **{post_book_metrics['sharpe_0rf']:.2f}**. Cela ne remet pas en cause l'orthogonalité, mais réduit la confiance dans la persistance de l'alpha et impose une phase shadow.

## Données et méthode

- Manuel : {manuel_raw.index.min().date()} à {manuel_raw.index.max().date()}, 4 sleeves et un total.
- Comparaison : {e_book.index.min().date()} à {e_book.index.max().date()}, calendrier de jours ouvrés, absences remplies à zéro.
- Le fichier Manuel contient {weekend_rows} lignes de week-end, dont {weekend_nonzero} non nulles ; elles sont exclues des métriques annualisées.
- Contrôle : erreur maximale entre `TOTAL` et la somme des 4 sleeves = {total_error:.4f}.
- Pour comparer des dollars/unités de tailles différentes, chaque sleeve est divisée par sa volatilité quotidienne sur la fenêtre commune. Les livres sont des moyennes equal-risk, puis remis à une volatilité unitaire pour les mélanges.
- Sharpe calculé avec 252 jours et taux sans risque nul. Il s'agit de PnL, pas de rendements sur capital : ne pas l'interpréter comme un rendement investisseur.

## Les 4 sleeves de Manuel — historique complet

{fmt_table(full_metrics, ['series','active_days','active_pct','total_pnl','annualized_pnl','annualized_vol','sharpe_0rf','max_drawdown','pnl_to_maxdd','active_hit_rate_pct','profit_factor'])}

![Courbes et drawdowns](01_manuel_equity_drawdown.png)

## Stabilité, ancienneté effective et concentration

La profondeur du fichier n'est pas la profondeur effective de chaque sleeve. `US30` et `AUS200` sont actifs dès 2019, tandis que `GER40` et `US100` sont très clairsemés avant 2021–2022. La dispersion annuelle des corrélations entre livres reste faible (minimum **{yearly_book_corr.min():.3f}**, maximum **{yearly_book_corr.max():.3f}**), mais les premières années ne testent donc pas les quatre moteurs de façon comparable.

**PnL annuel :**

{fmt_table(annual, list(annual.columns), 0)}

**Jours actifs annuels :**

{fmt_table(activity, list(activity.columns), 0)}

**Concentration et dates actives :**

{fmt_table(concentration, ['series','first_active_date','last_active_date','top_5_days_pct_total_pnl','top_10_days_pct_gross_profit','lag1_autocorrelation'])}

![Stabilité annuelle](06_yearly_stability.png)

La hausse du PnL brut à partir de 2022 peut refléter un meilleur régime, une augmentation de taille ou une modification du système. Sans série de risque/notionnel, le Sharpe et l'amélioration de portefeuille restent indicatifs. `GER40` est le plus concentré : ses cinq meilleurs jours représentent **{concentration.set_index('series').loc['GER40', 'top_5_days_pct_total_pnl']:.1f} %** de son PnL net.

## Pseudo-holdout après la fin des données du livre existant

Du **{post_metrics.start.iloc[0]}** au **{post_metrics.end.iloc[0]}**, les données Manuel continuent alors que les six sleeves de comparaison s'arrêtent. Cette tranche n'est pas intégrée aux corrélations ni au choix des poids :

{fmt_table(post_metrics, ['series','active_days','total_pnl','annualized_pnl','annualized_vol','sharpe_0rf','max_drawdown','profit_factor'])}

Le contraste est marqué : `GER40` et `US100` tiennent, mais `AUS200` efface une part importante de ses gains et `US30` se dégrade. Comme on ignore si les paramètres étaient gelés avant mars 2026, il faut parler de **pseudo-holdout**, pas de preuve OOS. C'est néanmoins l'argument le plus fort contre une allocation immédiate de taille normale.

## Comparaison sur la fenêtre commune

{fmt_table(overlap_metrics, ['series','active_days','active_pct','total_pnl','annualized_pnl','annualized_vol','sharpe_0rf','max_drawdown','pnl_to_maxdd'])}

![Livres equal-risk](04_books_equity_drawdown.png)

## Orthogonalité sleeve par sleeve

| Sleeve Manuel | Corrélation absolue moyenne vs les 6 | Corrélation absolue maximale vs une sleeve | R² des 6 sleeves | Variance non expliquée |
|---|---:|---:|---:|---:|
"""
    for name in MANUEL:
        u = uniqueness.set_index("manuel_sleeve").loc[name]
        text += f"| {name} | {mean_abs_by_m[name]:.3f} | {max_abs_by_m[name]:.3f} | {u.in_sample_r2_vs_six:.3f} | {u.unexplained_variance_pct:.1f} % |\n"

    text += f"""

Les corrélations sur les seuls jours où les deux sleeves sont actives figurent dans `pairwise_diagnostics.csv`. Elles sont plus bruyantes car certains couples ont peu d'observations simultanées. Le R² OOS de seconde moitié est un diagnostic de stabilité, pas un backtest de signal.

![Matrice de corrélation clusterisée](02_correlation_cluster.png)

## Stabilité temporelle et alignement des dates

- Corrélation roulante 63 jours : médiane **{rolling.median():.3f}**, 10e–90e percentiles **[{rolling.quantile(.1):.3f}, {rolling.quantile(.9):.3f}]**.
- Plus forte corrélation absolue sur les décalages -3 à +3 jours : **{best_lag.correlation:.3f}** avec un décalage Manuel de **{int(best_lag.manuel_shift_business_days)}** jour(s). Une pointe hors zéro signalerait un problème potentiel de convention de date ou un lead-lag à investiguer.

![Corrélation roulante](03_rolling_book_correlation.png)

## Valeur marginale dans le portefeuille

{fmt_table(blends, ['manuel_risk_budget_pct','sharpe_0rf','annualized_vol_risk_units','max_drawdown_risk_units','pnl_to_maxdd'])}

![Sharpe du mélange](05_blend_sharpe.png)

## Comportement en stress

{fmt_table(regimes, ['test','series','observations','mean_daily_risk_units','positive_days_pct'])}

Ce test répond à la question la plus utile pour la diversification : Manuel gagne-t-il, ou au moins reste-t-il neutre, pendant les pires jours du livre existant ? Les zéros sont conservés car l'absence de position pendant un stress constitue bien une propriété de diversification.

## Ce que les données permettent — et ne permettent pas — de conclure

Les éléments en faveur d'un alpha différent sont : faible corrélation des PnL, faible pouvoir explicatif linéaire des six sleeves existantes, stabilité raisonnable de cette corrélation dans le temps et amélioration du profil du portefeuille après égalisation du risque.

Cela ne prouve pas encore un mécanisme différent. `US30`, `GER40`, `US100` et `AUS200` ressemblent à des étiquettes d'instruments, pas à des descriptions de signaux. Plusieurs stratégies proches peuvent paraître décorrélées si elles interviennent à des heures différentes, ont beaucoup de jours sans trade, utilisent des tailles variables ou enregistrent le PnL sur des dates différentes.

## Protocole recommandé avant échange de signaux

1. Demander un export trade-level : timestamps avec fuseau, instrument/contrat, direction, taille, prix d'entrée/sortie, frais et identifiant de setup — sans exiger son code ni ses paramètres propriétaires.
2. Refaire la comparaison sur un notionnel ou un risque ex ante commun, net de coûts et slippage.
3. Geler les règles et observer 8 à 12 semaines en shadow/paper, sans recalibrer le poids sur cette période.
4. Vérifier les expositions communes : beta indices, heure de la journée, overnight/intraday, volatilité, trend/reversal, jours macro et concentration sur les dix meilleurs jours.
5. Si l'échange démarre, commencer à 10–20 % du budget de risque prévu, avec kill switch, limite de perte par sleeve et attribution quotidienne séparée.

## Fichiers produits

- `metrics_manuel_full.csv` et `metrics_overlap.csv`
- `correlation_matrix.csv` et `pairwise_diagnostics.csv`
- `regression_uniqueness.csv`, `regime_stress.csv`, `lead_lag.csv`
- `yearly_diagnostics.csv`, `concentration_diagnostics.csv`
- `post_overlap_metrics.csv` (pseudo-holdout Manuel uniquement)
- `portfolio_blends.csv`, `bootstrap_uncertainty.csv`
- `aligned_daily_pnl.csv` (données alignées utilisées pour l'analyse)
"""
    (out / "REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuel", type=Path, required=True)
    parser.add_argument("--emile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manuel_raw, emile_raw = load_inputs(args.manuel, args.emile)
    start = max(manuel_raw.index.min(), emile_raw.index.min())
    end = min(manuel_raw.index.max(), emile_raw.index.max())
    manuel_full = business_grid(manuel_raw[MANUEL], manuel_raw.index.min(), manuel_raw.index.max())
    manuel = business_grid(manuel_raw[MANUEL], start, end)
    emile = business_grid(emile_raw[EMILE], start, end)
    data = pd.concat([emile, manuel], axis=1)
    data_z = risk_normalize(data)
    e_book = data_z[EMILE].mean(axis=1)
    m_book = data_z[MANUEL].mean(axis=1)

    full_metrics = pd.DataFrame([{"series": c, **metrics(manuel_full[c])} for c in MANUEL])
    overlap_metrics = pd.DataFrame([{"series": c, **metrics(data[c])} for c in EMILE + MANUEL])
    pairwise = pair_diagnostics(data)
    corr = data.corr()
    uniqueness = regression_uniqueness(data_z)
    blends = blend_table(e_book, m_book)
    bootstrap = bootstrap_books(e_book, m_book)
    regimes = regime_table(data_z, e_book, m_book)
    leadlags = lead_lag(e_book, m_book)
    yearly = yearly_diagnostics(data, data_z)
    concentration = concentration_diagnostics(data)
    post_start = end + pd.offsets.BDay(1)
    post_metrics: pd.DataFrame
    post_book_metrics: dict[str, float] | None
    if post_start <= manuel_raw.index.max():
        manuel_post = business_grid(manuel_raw[MANUEL], post_start, manuel_raw.index.max())
        post_metrics = pd.DataFrame([{"series": c, **metrics(manuel_post[c])} for c in MANUEL])
        overlap_vol = data[MANUEL].std(ddof=1).replace(0, np.nan)
        post_book = manuel_post.divide(overlap_vol).mean(axis=1)
        post_book_metrics = metrics(post_book)
    else:
        post_metrics = pd.DataFrame()
        post_book_metrics = None

    data.to_csv(args.output / "aligned_daily_pnl.csv", index_label="date")
    full_metrics.to_csv(args.output / "metrics_manuel_full.csv", index=False)
    overlap_metrics.to_csv(args.output / "metrics_overlap.csv", index=False)
    corr.to_csv(args.output / "correlation_matrix.csv")
    pairwise.to_csv(args.output / "pairwise_diagnostics.csv", index=False)
    uniqueness.to_csv(args.output / "regression_uniqueness.csv", index=False)
    blends.to_csv(args.output / "portfolio_blends.csv", index=False)
    bootstrap.to_csv(args.output / "bootstrap_uncertainty.csv", index=False)
    regimes.to_csv(args.output / "regime_stress.csv", index=False)
    leadlags.to_csv(args.output / "lead_lag.csv", index=False)
    yearly.to_csv(args.output / "yearly_diagnostics.csv", index=False)
    concentration.to_csv(args.output / "concentration_diagnostics.csv", index=False)
    post_metrics.to_csv(args.output / "post_overlap_metrics.csv", index=False)

    save_figures(args.output, manuel_full, data_z, e_book, m_book, corr, blends, yearly)
    build_report(
        args.output,
        manuel_raw,
        full_metrics,
        overlap_metrics,
        pairwise,
        uniqueness,
        blends,
        bootstrap,
        regimes,
        leadlags,
        yearly,
        concentration,
        post_metrics,
        post_book_metrics,
        e_book,
        m_book,
    )


if __name__ == "__main__":
    main()
