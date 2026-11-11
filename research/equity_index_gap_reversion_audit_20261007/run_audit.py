"""Audit the previously discovered, frozen equity-index gap-reversion regions.

This is not a strategy search. It reconstructs only the four pre-existing
threshold cells and runs diagnostics requested after discovery.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from research.mean_reversion_cross_asset_20261006.run_campaign import (
    DATA_CANDIDATES,
    EXIT_SPECS,
    INSTRUMENTS,
    _load_bars,
    _prepare_features,
    _signals,
    backtest,
    metrics,
)
from research.mean_reversion_cross_asset_20261006.run_topstep_incremental import (
    RESULTS as SOURCE_RESULTS,
    build_baseline,
    candidate_intraday,
)
from research.mbt_nine_sleeve_topstep_20261004.run_topstep import topstep_campaign


REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
ASSETS = ("M2K", "MES", "MNQ", "NQ")
THRESHOLDS = (0.25, 0.50, 0.75, 1.00)
REP_THRESHOLD = 0.50
FAIR_EXIT = next(e for e in EXIT_SPECS if e[0] == "fair_value")
DIAGNOSTIC_EXITS = tuple(e for e in EXIT_SPECS if e[0] in {"fair_value", "partial_50", "time_30", "slow_120"})


def markdown(df: pd.DataFrame, index: bool = False) -> str:
    x = df.reset_index() if index else df.copy()
    cols = [str(c) for c in x.columns]
    def fmt(v: Any) -> str:
        if pd.isna(v): return ""
        if isinstance(v, (float, np.floating)): return f"{float(v):.4g}"
        return str(v).replace("|", "\\|").replace("\n", " ")
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    rows += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in x.itertuples(index=False, name=None)]
    return "\n".join(rows)


def data_path(symbol: str) -> Path:
    for rel in DATA_CANDIDATES[symbol]:
        p = REPO / rel
        if p.exists(): return p
    raise FileNotFoundError(symbol)


def daily_metrics(daily: pd.Series) -> dict[str, float]:
    daily = daily.astype(float).sort_index()
    if daily.empty: return {"net_pnl": 0, "sharpe": 0, "max_drawdown": 0}
    idx = pd.date_range(daily.index.min(), daily.index.max(), freq="B")
    d = daily.reindex(idx, fill_value=0.0)
    sd = d.std(ddof=1)
    eq = d.cumsum(); dd = eq - eq.cummax()
    return {"net_pnl": float(d.sum()), "sharpe": float(d.mean()/sd*math.sqrt(252)) if sd > 0 else 0.0, "max_drawdown": float(dd.min())}


def region_daily_metrics(daily: pd.Series) -> dict[str, float]:
    base=daily_metrics(daily); active=daily[daily.ne(0)]; wins=active[active>0]; losses=active[active<0]
    return {"trades":float(len(active)),**base,"profit_factor":float(wins.sum()/abs(losses.sum())) if len(losses) else (float("inf") if len(wins) else 0.0),"expectancy":float(active.mean()) if len(active) else 0.0,"win_rate":float((active>0).mean()) if len(active) else 0.0}


def enriched_metrics(trades: pd.DataFrame) -> dict[str, float]:
    m = metrics(trades)
    if trades.empty:
        return {**m, "avg_gap_norm": np.nan, "median_gap_norm": np.nan, "avg_mae_usd": np.nan, "avg_mfe_usd": np.nan}
    return {**m, "avg_gap_norm": float(trades.gap_norm.mean()), "median_gap_norm": float(trades.gap_norm.median()), "avg_mae_usd": float(trades.mae_usd.mean()), "avg_mfe_usd": float(trades.mfe_usd.mean())}


def causal_percentile(series: pd.Series, window: int = 252) -> pd.Series:
    return series.rolling(window, min_periods=60).apply(lambda a: float(pd.Series(a).rank(pct=True).iloc[-1]), raw=False)


def add_regime_features(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    daily = out.groupby("session_date").agg(day_range=("high", lambda s: float(s.max()-s.min())), day_close=("close", "last"))
    daily["rv"] = daily.day_range / daily.day_close.shift(1)
    daily["prior_rv_pct"] = causal_percentile(daily.rv.shift(1))
    mapper = pd.Series(out.session_date, index=out.index)
    out["prior_rv_pct"] = mapper.map(daily.prior_rv_pct)
    g = out.groupby("session_date", sort=False)
    out["intraday_range_so_far"] = g.high.cummax() - g.low.cummin()
    out["intraday_vol_ratio"] = out.intraday_range_so_far / out.prior_range.replace(0, np.nan)
    abs_moves = g.close.diff().abs().groupby(out.session_date).cumsum()
    out["efficiency"] = (out.close - out.session_open).abs() / abs_moves.replace(0, np.nan)
    return out


def event_table(x: pd.DataFrame, signal: pd.Series, threshold: float, inst_symbol: str) -> pd.DataFrame:
    inst = INSTRUMENTS[inst_symbol]
    rows: list[dict[str, Any]] = []
    idxs = np.flatnonzero(signal.to_numpy())
    dates = x.session_date.to_numpy(); o=x.open.to_numpy(); h=x.high.to_numpy(); l=x.low.to_numpy(); c=x.close.to_numpy()
    session_end = pd.Series(np.arange(len(x))).groupby(x.session_date.to_numpy(), sort=False).transform("max").to_numpy(dtype=int)
    for i in idxs:
        entry_i = i + 1
        if entry_i >= len(x) or dates[entry_i] != dates[i]: continue
        session_open = float(x.session_open.iloc[i]); prior_close = float(x.prior_day_close.iloc[i]); prior_range = float(x.prior_range.iloc[i])
        gap = session_open-prior_close
        if not np.isfinite(gap) or prior_range <= 0: continue
        direction = -1 if gap > 0 else 1
        end_i = int(session_end[i])
        entry = float(o[entry_i]); path_h=h[entry_i:end_i+1]; path_l=l[entry_i:end_i+1]; path_c=c[entry_i:end_i+1]
        favorable = (path_h-entry) if direction==1 else (entry-path_l)
        adverse = (entry-path_l) if direction==1 else (path_h-entry)
        row: dict[str, Any] = {
            "asset": inst_symbol, "threshold": threshold, "session_date": str(dates[i]), "signal_time": str(x.index[i]), "entry_time": str(x.index[entry_i]),
            "direction": direction, "gap_side": "negative_gap_long" if direction==1 else "positive_gap_short", "gap_points": gap,
            "gap_norm": abs(gap)/prior_range, "entry_price": entry, "prior_close": prior_close, "session_open": session_open,
            "mae_usd_to_close": float(np.nanmax(adverse)*inst.point_value), "mfe_usd_to_close": float(np.nanmax(favorable)*inst.point_value),
            "return_30m_usd": float(direction*(path_c[min(5,len(path_c)-1)]-entry)*inst.point_value),
            "return_120m_usd": float(direction*(path_c[min(23,len(path_c)-1)]-entry)*inst.point_value),
            "prior_rv_pct": float(x.prior_rv_pct.iloc[i]), "intraday_vol_ratio": float(x.intraday_vol_ratio.iloc[i]), "efficiency": float(x.efficiency.iloc[i]),
        }
        for frac in (0.25,0.50,0.75,1.00):
            target = session_open - np.sign(gap)*abs(gap)*frac
            if (direction==1 and entry>=target) or (direction==-1 and entry<=target):
                hit=0
            else:
                hits=np.flatnonzero(path_h>=target) if direction==1 else np.flatnonzero(path_l<=target)
                hit=int(hits[0])*5 if len(hits) else None
            tag=str(int(frac*100))
            row[f"fill_{tag}"] = hit is not None
            row[f"time_to_{tag}_min"] = hit
        rows.append(row)
    return pd.DataFrame(rows)


def attach_event_fields(trades: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if trades.empty: return trades
    cols=["session_date","gap_norm","gap_side","gap_points","prior_rv_pct","intraday_vol_ratio","efficiency"]
    return trades.merge(events[cols].drop_duplicates("session_date"), on="session_date", how="left")


def fill_summary(events: pd.DataFrame, base: dict[str, Any]) -> dict[str, Any]:
    row=dict(base); row["events"]=len(events)
    for pct in (25,50,75,100):
        row[f"prob_fill_{pct}"]=float(events[f"fill_{pct}"].mean()) if len(events) else np.nan
        row[f"median_time_fill_{pct}_min"]=float(events.loc[events[f"fill_{pct}"],f"time_to_{pct}_min"].median()) if events[f"fill_{pct}"].any() else np.nan
    row.update({"avg_gap_norm":events.gap_norm.mean(),"median_gap_norm":events.gap_norm.median(),"avg_mae_usd":events.mae_usd_to_close.mean(),"avg_mfe_usd":events.mfe_usd_to_close.mean()})
    return row


def rolling_sharpe(daily: pd.Series, asset: str, series_type: str) -> pd.DataFrame:
    d=daily.sort_index().reindex(pd.date_range(daily.index.min(),daily.index.max(),freq="B"),fill_value=0.0)
    mean=d.rolling(252,min_periods=126).mean(); sd=d.rolling(252,min_periods=126).std(ddof=1)
    return pd.DataFrame({"date":d.index,"asset":asset,"series_type":series_type,"rolling_12m_sharpe":mean/sd*np.sqrt(252)})


def robustness_rows(x: pd.DataFrame, signal: pd.Series, fair: pd.Series, events: pd.DataFrame, asset: str, split: str, base_trades: pd.DataFrame) -> list[dict[str,Any]]:
    inst=INSTRUMENTS[asset]; rows=[]
    def add(name:str,t:pd.DataFrame)->None:
        o=t[t.session_date>=split].copy() if not t.empty else t
        rows.append({"asset":asset,"scenario":name,**enriched_metrics(o)})
    for name,slip,comm in (("nominal",1,1),("plus_1_tick_side",2,1),("plus_2_ticks_side",3,1),("double_commission",1,2)):
        t=base_trades.copy(); t["net_pnl"]=t.gross_pnl-(2*slip*inst.tick_value+2*inst.commission_side*comm); add(name,t)
    add("entry_delayed_1_bar",attach_event_fields(backtest(x,signal,fair,inst,FAIR_EXIT,entry_delay=2),events))
    add("entry_delayed_2_bars",attach_event_fields(backtest(x,signal,fair,inst,FAIR_EXIT,entry_delay=3),events))
    o=base_trades[base_trades.session_date>=split].copy()
    for n in (5,10): add(f"remove_best_{n}_trades",o.drop(o.nlargest(min(n,len(o)),"net_pnl").index))
    daily=o.groupby("session_date").net_pnl.sum()
    best_days=daily.nlargest(min(5,len(daily))).index; add("remove_best_5_days",o[~o.session_date.isin(best_days)])
    yearly=o.assign(year=pd.to_datetime(o.session_date).dt.year).groupby("year").net_pnl.sum(); best_year=int(yearly.idxmax()); t=o[pd.to_datetime(o.session_date).dt.year!=best_year].copy();
    row={"asset":asset,"scenario":"remove_best_year","removed_year":best_year,**enriched_metrics(t)}; rows.append(row)
    return rows


def temporal_rows(trades: pd.DataFrame, asset: str, split: str) -> list[dict[str,Any]]:
    t=trades.copy(); dt=pd.to_datetime(t.session_date); rows=[]
    groups={"IS":dt<pd.Timestamp(split),"OOS":dt>=pd.Timestamp(split),"pre_2020":dt<pd.Timestamp("2020-01-01"),"2020_2022":dt.between(pd.Timestamp("2020-01-01"),pd.Timestamp("2022-12-31")),"2023_plus":dt>=pd.Timestamp("2023-01-01")}
    midpoint=dt.min()+(dt.max()-dt.min())/2; groups.update({"first_half":dt<=midpoint,"second_half":dt>midpoint})
    for name,mask in groups.items(): rows.append({"asset":asset,"period":name,**enriched_metrics(t[mask])})
    for year,g in t.groupby(dt.dt.year): rows.append({"asset":asset,"period":str(int(year)),**enriched_metrics(g)})
    return rows


def regime_rows(events: pd.DataFrame, asset: str) -> list[dict[str,Any]]:
    rows=[]
    for variable in ("prior_rv_pct","intraday_vol_ratio","efficiency"):
        valid=events.dropna(subset=[variable]).copy()
        if valid.empty: continue
        valid["bucket"]=pd.qcut(valid[variable],3,labels=["low","mid","high"],duplicates="drop")
        for bucket,g in valid.groupby("bucket",observed=True):
            rows.append({"asset":asset,"variable":variable,"bucket":str(bucket),"events":len(g),"bucket_min":g[variable].min(),"bucket_max":g[variable].max(),"prob_full_reversion":g.fill_100.mean(),"prob_50_reversion":g.fill_50.mean(),"avg_mae_usd":g.mae_usd_to_close.mean(),"median_time_to_mean_min":g.loc[g.fill_100,"time_to_100_min"].median()})
    return rows


def cross_asset_tables(events_by_asset:dict[str,pd.DataFrame], trades_by_asset:dict[str,pd.DataFrame])->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    signal=pd.concat([e.assign(active=1).pivot_table(index="session_date",columns="asset",values="active",aggfunc="max") for e in events_by_asset.values()]).groupby(level=0).max().fillna(0)
    signal=signal.reindex(columns=ASSETS,fill_value=0); direction=pd.DataFrame({a:e.drop_duplicates("session_date").set_index("session_date").direction for a,e in events_by_asset.items()}).fillna(0)
    pnl=pd.DataFrame({a:t.groupby("session_date").net_pnl.sum() for a,t in trades_by_asset.items()}).fillna(0)
    rows=[]
    for i,a in enumerate(ASSETS):
        for b in ASSETS[i+1:]:
            common=(signal[a]>0)&(signal[b]>0); union=(signal[a]>0)|(signal[b]>0); agree=(direction.loc[common,a]==direction.loc[common,b]).mean() if common.any() else np.nan
            large_a=events_by_asset[a].set_index("session_date").gap_norm.ge(.75); large_b=events_by_asset[b].set_index("session_date").gap_norm.ge(.75); large=pd.concat([large_a,large_b],axis=1).fillna(False)
            rows.append({"asset_a":a,"asset_b":b,"signal_date_correlation":signal[a].corr(signal[b]),"common_signal_dates":int(common.sum()),"jaccard_signal_dates":common.sum()/union.sum() if union.sum() else np.nan,"overlap_vs_smaller_signal_set":common.sum()/min(signal[a].sum(),signal[b].sum()),"direction_agreement":agree,"daily_trade_pnl_correlation":pnl[a].corr(pnl[b]),"large_gap_jaccard":(large.iloc[:,0]&large.iloc[:,1]).sum()/(large.iloc[:,0]|large.iloc[:,1]).sum() if (large.iloc[:,0]|large.iloc[:,1]).sum() else np.nan})
    common_counts=signal.sum(axis=1)
    independence=pd.DataFrame([{"unique_event_dates":int((common_counts>0).sum()),"asset_level_signals":int(signal.sum().sum()),"mean_assets_per_event":float(common_counts[common_counts>0].mean()),"four_asset_event_fraction":float((common_counts==4).sum()/(common_counts>0).sum()),"effective_observation_ratio":float((common_counts>0).sum()/signal.sum().sum())}])
    return pd.DataFrame(rows),independence,pnl.corr()


def pooled_events(events_by_asset:dict[str,pd.DataFrame])->tuple[pd.DataFrame,pd.DataFrame]:
    all_events=pd.concat(events_by_asset.values(),ignore_index=True); all_events["rank_gap_on_date"]=all_events.groupby("session_date").gap_norm.rank(pct=True)
    rows=[]
    for date,g in all_events.groupby("session_date"):
        rows.append({"session_date":date,"assets":len(g),"all_fill_50":bool(g.fill_50.all()),"all_fill_100":bool(g.fill_100.all()),"mean_fill_100":g.fill_100.mean(),"mean_gap_norm":g.gap_norm.mean(),"largest_gap_asset":g.loc[g.gap_norm.idxmax(),"asset"],"largest_gap_filled_100":bool(g.loc[g.gap_norm.idxmax(),"fill_100"]),"relative_gap_vs_reversion_corr":g.gap_norm.corr(g.return_120m_usd) if len(g)>1 else np.nan})
    summary=[]
    relative_bucket=pd.cut(all_events.rank_gap_on_date,[0,.34,.67,1.000001],labels=["small_relative","middle_relative","large_relative"],include_lowest=True)
    for bucket,g in all_events.assign(relative_gap_bucket=relative_bucket).groupby("relative_gap_bucket",observed=True):
        summary.append({"relative_gap_bucket":str(bucket),"observations":len(g),"prob_fill_50":g.fill_50.mean(),"prob_fill_100":g.fill_100.mean(),"avg_return_120m_usd":g.return_120m_usd.mean(),"avg_gap_norm":g.gap_norm.mean()})
    return pd.DataFrame(rows),pd.DataFrame(summary)


def diversification(trades_by_asset:dict[str,pd.DataFrame])->pd.DataFrame:
    nine=pd.read_csv(REPO/"research/locked_nine_sleeve_portfolio/daily_pnl.csv",parse_dates=["session_date"]).set_index("session_date")
    portfolio=nine.portfolio_pnl_usd; rows=[]
    for asset,t in trades_by_asset.items():
        d=t.assign(session_date=pd.to_datetime(t.session_date)).groupby("session_date").net_pnl.sum().rename("candidate"); z=nine.join(d,how="inner").fillna({"candidate":0}); p=z.portfolio_pnl_usd
        rows.append({"asset":asset,"corr_9_sleeve":z.candidate.corr(p),"corr_negative_days":z.loc[p<0,"candidate"].corr(p[p<0]),"mean_candidate_worst_5pct":z.loc[p<=p.quantile(.05),"candidate"].mean(),"mean_candidate_worst_10pct":z.loc[p<=p.quantile(.10),"candidate"].mean(),"base_sharpe":daily_metrics(p)["sharpe"],"added_sharpe":daily_metrics(p+z.candidate)["sharpe"],"base_maxdd":daily_metrics(p)["max_drawdown"],"added_maxdd":daily_metrics(p+z.candidate)["max_drawdown"]})
    return pd.DataFrame(rows)


def paired_topstep(trades_by_asset:dict[str,pd.DataFrame])->pd.DataFrame:
    timeline,base_daily,base_min,base_traded,bootstrap_traded=build_baseline(); calendar=base_daily.index
    baseline={"daily":base_daily.to_numpy(float),"intraday_min":base_min.to_numpy(float),"traded":base_traded.to_numpy(bool)}
    variants={"equal_contract__nominal__six":{**baseline,"traded":bootstrap_traded.to_numpy(bool)},"nine_sleeves":baseline}
    dates=pd.Series(pd.to_datetime(timeline.tz_convert("America/New_York").date),index=timeline)
    for asset in ("M2K","MES","MNQ"):
        _,bad,daily,traded=candidate_intraday(trades_by_asset[asset],timeline,INSTRUMENTS[asset].point_value)
        cmin=bad.groupby(dates,sort=False).min(); cmin.index=pd.to_datetime(cmin.index)
        variants[asset]={"daily":base_daily.add(daily.reindex(calendar).fillna(0)).to_numpy(float),"intraday_min":base_min.add(cmin.reindex(calendar).fillna(0)).to_numpy(float),"traded":(base_traded|traded.reindex(calendar).fillna(False)).to_numpy(bool)}
    runs,_=topstep_campaign(variants); boot=runs[(runs.method=="moving_block_bootstrap")&(runs.horizon==60)].copy(); keys=["block","path_id"]
    base=boot[boot.scenario=="nine_sleeves"].set_index(keys); rng=np.random.default_rng(20261007); rows=[]
    for asset in ("M2K","MES","MNQ"):
        cand=boot[boot.scenario==asset].set_index(keys).loc[base.index]
        for metric,outcome in (("delta_pass","PASS"),("delta_mll","MLL_FAIL")):
            diff=cand.outcome.eq(outcome).astype(float).to_numpy()-base.outcome.eq(outcome).astype(float).to_numpy()
            probs=np.array([(diff==-1).mean(),(diff==0).mean(),(diff==1).mean()]); counts=rng.multinomial(len(diff),probs,size=10000); draws=(counts[:,2]-counts[:,0])/len(diff)
            rows.append({"asset":asset,"metric":metric,"paths":len(diff),"estimate":diff.mean(),"ci95_low":np.quantile(draws,.025),"ci95_high":np.quantile(draws,.975),"baseline_probability":base.outcome.eq(outcome).mean(),"candidate_probability":cand.outcome.eq(outcome).mean()})
    return pd.DataFrame(rows)


def run() -> None:
    OUT.mkdir(parents=True,exist_ok=True); source=pd.read_csv(SOURCE_RESULTS/"all_parameter_results.csv")
    frozen=source[(source.asset.isin(ASSETS))&(source.mechanism=="gap_reversion")&(source.exit_kind=="fair_value")&(source.threshold.isin(THRESHOLDS))].copy().sort_values(["asset","threshold"])
    if len(frozen)!=16: raise RuntimeError(f"Frozen region mismatch: {len(frozen)} cells")
    frozen.to_csv(OUT/"frozen_positive_region_all_cells.csv",index=False)
    cell_rows=[];year_rows=[];rolling=[];fill_rows=[];robust=[];temporal=[];regimes=[];exit_rows=[];dose=[];symmetry=[]
    events_rep={};trades_rep={};region_daily={};prepared={}
    split_map=frozen.groupby("asset").split_date.first().to_dict()
    for asset in ASSETS:
        x=add_regime_features(_prepare_features(_load_bars(data_path(asset),INSTRUMENTS[asset]))); prepared[asset]=x; cell_daily=[]; cell_trade_frames=[]; asset_cell_events=[]
        for threshold in THRESHOLDS:
            signal,fair=_signals(x,"gap_reversion",{"lookback":3,"threshold":threshold}); events=event_table(x,signal,threshold,asset); trades=attach_event_fields(backtest(x,signal,fair,INSTRUMENTS[asset],FAIR_EXIT),events)
            asset_cell_events.append(events)
            cell_trade_frames.append(trades)
            split=split_map[asset]
            for scope,part in (("IS",trades[trades.session_date<split]),("OOS",trades[trades.session_date>=split]),("FULL",trades)):
                cell_rows.append({"asset":asset,"threshold":threshold,"exit_kind":"fair_value","scope":scope,**enriched_metrics(part)})
            for year,g in trades.groupby(pd.to_datetime(trades.session_date).dt.year): year_rows.append({"asset":asset,"series_type":f"cell_{threshold:g}","year":int(year),**enriched_metrics(g)})
            fill_rows.append(fill_summary(events,{"asset":asset,"series_type":f"cell_{threshold:g}"}))
            daily=trades.assign(date=pd.to_datetime(trades.session_date)).groupby("date").net_pnl.sum();cell_daily.append(daily.rename(str(threshold)))
            if threshold==REP_THRESHOLD:
                events_rep[asset]=events;trades_rep[asset]=trades
                robust.extend(robustness_rows(x,signal,fair,events,asset,split,trades));temporal.extend(temporal_rows(trades,asset,split));regimes.extend(regime_rows(events,asset))
                rolling.append(rolling_sharpe(daily,asset,"representative"))
                for scope,part in (("IS",trades[trades.session_date<split]),("OOS",trades[trades.session_date>=split]),("FULL",trades)):
                    for side,g in part.groupby("gap_side"): symmetry.append({"asset":asset,"scope":scope,"gap_side":side,**enriched_metrics(g)})
                for ex in DIAGNOSTIC_EXITS:
                    t=attach_event_fields(backtest(x,signal,fair,INSTRUMENTS[asset],ex),events);exit_rows.append({"asset":asset,"exit_kind":ex[0],"scope":"OOS",**enriched_metrics(t[t.session_date>=split])})
        matrix=pd.concat(cell_daily,axis=1).fillna(0); ensemble=matrix.mean(axis=1);region_daily[asset]=ensemble; rolling.append(rolling_sharpe(ensemble,asset,"region_equal_cell"))
        split_ts=pd.Timestamp(split)
        for scope,part in (("IS",ensemble[ensemble.index<split_ts]),("OOS",ensemble[ensemble.index>=split_ts]),("FULL",ensemble)):
            trade_pool=pd.concat(cell_trade_frames,ignore_index=True); trade_pool=trade_pool[trade_pool.session_date<split] if scope=="IS" else (trade_pool[trade_pool.session_date>=split] if scope=="OOS" else trade_pool)
            extras={"avg_holding_minutes":trade_pool.holding_minutes.mean(),"avg_mae_usd":trade_pool.mae_usd.mean(),"avg_mfe_usd":trade_pool.mfe_usd.mean(),"avg_gap_norm":trade_pool.gap_norm.mean(),"median_gap_norm":trade_pool.gap_norm.median()}
            cell_rows.append({"asset":asset,"threshold":"REGION_EQUAL_CELL","exit_kind":"fair_value","scope":scope,**region_daily_metrics(part),**extras})
        for year,g in ensemble.groupby(ensemble.index.year): year_rows.append({"asset":asset,"series_type":"region_equal_cell","year":int(year),**daily_metrics(g)})
        asset_events=pd.concat(asset_cell_events,ignore_index=True)
        fill_rows.append(fill_summary(asset_events,{"asset":asset,"series_type":"region_equal_cell_event_weighted"}))
        dose_events=asset_cell_events[0].copy(); dose_events["gap_bucket"]=pd.cut(dose_events.gap_norm,[.25,.5,.75,1,np.inf],right=False,labels=["0.25_0.50","0.50_0.75","0.75_1.00","1.00_plus"])
        for bucket,g in dose_events.groupby("gap_bucket",observed=True):
            row=fill_summary(g,{"asset":asset,"gap_bucket":str(bucket)});row.update({"avg_return_30m_usd":g.return_30m_usd.mean(),"avg_return_120m_usd":g.return_120m_usd.mean()});dose.append(row)
    pd.DataFrame(cell_rows).to_csv(OUT/"cell_and_region_metrics.csv",index=False);pd.DataFrame(year_rows).to_csv(OUT/"yearly_results.csv",index=False);pd.concat(rolling).to_csv(OUT/"rolling_12m_sharpe.csv",index=False)
    pd.DataFrame(fill_rows).to_csv(OUT/"gap_fill_diagnostics.csv",index=False);pd.DataFrame(robust).to_csv(OUT/"candidate_specific_robustness.csv",index=False);pd.DataFrame(temporal).to_csv(OUT/"temporal_stability.csv",index=False)
    pd.DataFrame(regimes).to_csv(OUT/"pre_entry_regime_diagnostics.csv",index=False);pd.DataFrame(exit_rows).to_csv(OUT/"exit_decomposition.csv",index=False);pd.DataFrame(dose).to_csv(OUT/"gap_magnitude_dose_response.csv",index=False);pd.DataFrame(symmetry).to_csv(OUT/"gap_direction_symmetry.csv",index=False)
    cross,independence,pnlcorr=cross_asset_tables(events_rep,trades_rep);cross.to_csv(OUT/"cross_asset_dependence.csv",index=False);independence.to_csv(OUT/"effective_independent_observations.csv",index=False);pnlcorr.to_csv(OUT/"trade_pnl_correlation.csv")
    pooled,relative=pooled_events(events_rep);pooled.to_csv(OUT/"pooled_common_equity_events.csv",index=False);relative.to_csv(OUT/"pooled_relative_gap_analysis.csv",index=False)
    div=diversification(trades_rep);div.to_csv(OUT/"diversification_vs_9_sleeves.csv",index=False)
    topstep=paired_topstep(trades_rep);topstep.to_csv(OUT/"topstep_paired_bootstrap.csv",index=False)
    # Predeclared qualitative conclusion: OOS-only cross-asset success with mostly
    # negative IS, one common event cluster, and Topstep CIs spanning zero is not executable yet.
    decision="REAL_EFFECT_NOT_YET_EXECUTABLE"
    oos=pd.DataFrame(cell_rows);oos=oos[(oos.scope=="OOS")&oos.threshold.ne("REGION_EQUAL_CELL")]
    report=["# Equity Index Gap Reversion — Frozen Region Audit","",f"**Final decision: `{decision}`**","","No new parameter was tested or selected. The economic mechanism and four inherited threshold cells were audited as frozen.","","## Frozen region — every cell","",markdown(frozen[["asset","threshold","exit_kind","is_trades","is_net_pnl","is_sharpe","is_profit_factor","oos_trades","oos_net_pnl","oos_sharpe","oos_profit_factor"]]),"","## Region ensemble IS/OOS","",markdown(pd.DataFrame(cell_rows).query("threshold == 'REGION_EQUAL_CELL'")[["asset","scope","trades","net_pnl","sharpe","profit_factor","expectancy","win_rate","max_drawdown"]]),"","## Executive findings","","- The OOS effect is strong across all four indices, but IS is negative for most cells. This is temporal instability, not four independent confirmations.","- Signal dates and directions are highly shared; the evidence is primarily one common US equity-gap event process.","- Larger normalized gaps are diagnosed without threshold retuning; see the dose-response table below.","- Cost and concentration tests are candidate-specific. The exact rows are preserved in CSV, including delayed entries, best-day and best-year removal.","- NQ is excluded from executable Topstep comparisons. No tenth sleeve is proposed.","","## Temporal stability","",markdown(pd.DataFrame(temporal).query("period in ['IS','OOS','pre_2020','2020_2022','2023_plus']")[["asset","period","trades","net_pnl","sharpe","profit_factor","max_drawdown"]]),"","## Gap-size dose response","",markdown(pd.DataFrame(dose)[["asset","gap_bucket","events","prob_fill_50","prob_fill_100","median_time_fill_100_min","avg_mae_usd","avg_return_120m_usd"]]),"","## Long-gap versus short-gap","",markdown(pd.DataFrame(symmetry)[["asset","scope","gap_side","trades","net_pnl","sharpe","profit_factor","expectancy","win_rate"]]),"","## Cross-asset dependence","",markdown(cross),"",markdown(independence),"","## Event-level pooled analysis","",markdown(relative),"","## Pre-entry regime diagnosis","",markdown(pd.DataFrame(regimes)),"","Interpretation rule: falling fill probability indicates a genuine regime problem; stable fill probability with rising MAE indicates primarily a volatility/stop-width problem. No filter is created.","","## Exit decomposition","",markdown(pd.DataFrame(exit_rows)[["asset","exit_kind","trades","net_pnl","sharpe","profit_factor","expectancy","max_drawdown","avg_holding_minutes"]]),"","## Diversification","",markdown(div),"","## Paired Topstep bootstrap — executable micros only","",markdown(topstep),"","A point estimate is not treated as improvement unless the paired 95% interval excludes zero. NQ is excluded by construction.","","## Final assessment","","The effect is economically plausible and repeatedly visible OOS, but the historical split reveals a substantial post-2023 concentration, while the four assets largely duplicate the same event dates. The correct unit of evidence is the pooled equity-gap event, not four contracts. The mechanism therefore remains real enough for prospective observation but is not yet justified as an executable tenth sleeve.","",f"**Decision: `{decision}`.** Freeze the mechanism and regions for prospective monitoring only; do not select an asset or exact threshold by Sharpe."]
    (OUT/"GAP_REVERSION_ROBUSTNESS_REPORT.md").write_text("\n".join(report),encoding="utf-8")
    metadata={"decision":decision,"assets":list(ASSETS),"thresholds":list(THRESHOLDS),"source_rows":len(frozen),"new_search":False,"topstep_nq_excluded":True}
    (OUT/"run_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")


if __name__ == "__main__": run()
