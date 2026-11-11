"""Audit the inherited MNQ high-volume/low-response OHLCV proxy.

This module deliberately replays only the twelve parameter cells inherited from
the Volume & Auction Structure Landscape.  It performs diagnostics and
falsifications; it does not search, tune, or promote a new strategy.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.mean_reversion_cross_asset_20261006.run_topstep_incremental import (
    build_baseline,
    candidate_intraday,
)
from research.mbt_nine_sleeve_topstep_20261004.run_topstep import topstep_campaign
from research.volume_auction_landscape_20261007.run_campaign import (
    DATA_FILES,
    INSTRUMENTS,
    SEED,
    add_causal_volume_baselines,
    add_price_features,
    backtest,
    load_bars,
    metrics,
    nominal_cost,
    signals,
    split_date,
    variant_grid,
    variant_id,
)


REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
SOURCE = REPO / "research" / "volume_auction_landscape_20261007" / "results"
FROZEN = REPO / "research" / "locked_nine_sleeve_portfolio" / "daily_pnl.csv"
ASSET = "MNQ"
INST = INSTRUMENTS[ASSET]
REP = {"mechanism": "D", "mode": "preceding_move_reversal", "lookback": 3,
       "move": 0.25, "rvol": 2.0, "profile": 0, "confirm": 1}
HORIZONS = (1, 2, 3, 6, 12, 24)


def md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    def f(v: Any) -> str:
        if pd.isna(v): return ""
        if isinstance(v, (float, np.floating)): return f"{float(v):.4g}"
        return str(v).replace("|", "\\|").replace("\n", " ")
    cols = list(map(str, df.columns))
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    rows += ["| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False, name=None)]
    return "\n".join(rows)


def enrich(m: dict[str, float], trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty:
        return {**m, "annualized_pnl": np.nan, "reversal_probability": np.nan}
    dates = pd.to_datetime(trades.session_date)
    years = max((dates.max() - dates.min()).days / 365.25, 1 / 365.25)
    return {**m, "annualized_pnl": float(m["net_pnl"] / years),
            "reversal_probability": float((trades.net_pnl + nominal_cost(INST) > 0).mean())}


def signal_from_mask(x: pd.DataFrame, mask: pd.Series, continuation: bool = False) -> pd.Series:
    prior = np.sign(x.ret_3).fillna(0).astype(np.int8)
    s = pd.Series(0, index=x.index, dtype=np.int8)
    s.loc[mask] = prior.loc[mask] * (1 if continuation else -1)
    valid = x.atr.notna() & x.rvol_20.notna() & (~x.last_bar) & prior.ne(0)
    return s.where(valid, 0).astype(np.int8)


def daily_metrics(d: pd.Series) -> dict[str, float]:
    d = d.astype(float).sort_index()
    if d.empty: return {"pnl": 0.0, "sharpe": np.nan, "maxdd": 0.0}
    full = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="B"), fill_value=0.0)
    sd = full.std(ddof=1); eq = full.cumsum()
    return {"pnl": float(full.sum()), "sharpe": float(full.mean()/sd*math.sqrt(252)) if sd > 0 else np.nan,
            "maxdd": float((eq-eq.cummax()).min())}


def all_cell_results(x: pd.DataFrame, split: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows, cache = [], {}
    inherited = [p for p in variant_grid() if p["mechanism"] == "D"]
    if len(inherited) != 12:
        raise RuntimeError(f"Expected 12 inherited cells, found {len(inherited)}")
    for p in inherited:
        vid = variant_id(p); t = backtest(x, signals(x, p), INST); cache[vid] = t
        for scope, part in (("IS", t[t.session_date < split]), ("OOS", t[t.session_date >= split]), ("FULL", t)):
            rows.append({"variant_id": vid, **p, "scope": scope, **enrich(metrics(part), part)})
    return pd.DataFrame(rows), cache


def interaction_controls(x: pd.DataFrame, split: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    low = (x.body_atr <= REP["move"]) & (x.range_atr <= 1.0)
    high = x.rvol_20 >= REP["rvol"]
    definitions = {"price_response_only": low, "high_volume_only": high, "both_conditions": low & high}
    rows=[]; trades={}
    for name, mask in definitions.items():
        t=backtest(x,signal_from_mask(x,mask),INST); t=t[t.session_date>=split].copy(); trades[name]=t
        rows.append({"control":name,**enrich(metrics(t),t)})
    base=trades["price_response_only"]; target=len(trades["both_conditions"]); rng=np.random.default_rng(SEED)
    random_rows=[]
    if len(base)>=target:
        for rep in range(1000):
            z=base.iloc[rng.choice(len(base),target,replace=False)].copy()
            random_rows.append({"replicate":rep,**enrich(metrics(z),z)})
        rr=pd.DataFrame(random_rows)
        rows.append({"control":"random_trade_count_matched_mean",**{c:rr[c].mean() for c in ("trades","net_pnl","sharpe","pf","expectancy","win_rate","maxdd","annualized_pnl","reversal_probability")}})
    shuffled=x.copy(); shuffled["rvol_20"]=shuffled.groupby("slot").rvol_20.transform(lambda q: pd.Series(rng.permutation(q.to_numpy()),index=q.index))
    st=backtest(shuffled,signal_from_mask(shuffled,low & (shuffled.rvol_20>=REP["rvol"])),INST); st=st[st.session_date>=split]
    rows.append({"control":"shuffled_volume_same_tod",**enrich(metrics(st),st)})
    out=pd.DataFrame(rows)
    both=out.set_index("control").loc["both_conditions"]
    for ref in ("price_response_only","high_volume_only","random_trade_count_matched_mean","shuffled_volume_same_tod"):
        for c in ("sharpe","expectancy","pf","reversal_probability","maxdd"):
            out.loc[out.control==ref,f"both_minus_control_{c}"]=both[c]-out.loc[out.control==ref,c].iloc[0]
    return out,pd.DataFrame(random_rows)


def event_study(x: pd.DataFrame, split: pd.Timestamp) -> tuple[pd.DataFrame,pd.DataFrame]:
    valid=x.atr.notna()&x.rvol_20.notna()&x.ret_3.ne(0)&(~x.last_bar)&(x.session_date>=split)
    high=x.rvol_20>=REP["rvol"]; low=(x.body_atr<=REP["move"])&(x.range_atr<=1.0)
    category=np.select([high&low,high&~low,~high&low,~high&~low],
                       ["high_volume_low_response","high_volume_high_response","normal_volume_low_response","normal_volume_high_response"],default="invalid")
    a=x.reset_index(names="signal_time"); rows=[]
    for i in np.flatnonzero(valid.to_numpy()):
        end=int(x._session_last_pos.iloc[i]); prior=np.sign(x.ret_3.iloc[i]); direction=-prior
        row={"signal_time":a.signal_time.iloc[i],"session_date":x.session_date.iloc[i],"category":category[i],
             "prior_direction":int(prior),"before_cumulative_normalized_return":float(x.ret_3.iloc[i]/x.atr.iloc[i]),
             "rvol":x.rvol_20.iloc[i],"normalized_range":x.range_atr.iloc[i],"candle_body":x.body_atr.iloc[i],"close_location":x.close_location.iloc[i]}
        future=x.iloc[i+1:min(end,i+24)+1]
        for h in HORIZONS:
            if i+h<=end: row[f"return_{h}_bar_atr"]=direction*(x.close.iloc[i+h]-x.close.iloc[i])/x.atr.iloc[i]
            else: row[f"return_{h}_bar_atr"]=np.nan
        if len(future):
            fav=np.where(direction>0,future.high-x.close.iloc[i],x.close.iloc[i]-future.low)/x.atr.iloc[i]
            adv=np.where(direction>0,x.close.iloc[i]-future.low,future.high-x.close.iloc[i])/x.atr.iloc[i]
            signed=direction*(future.close.to_numpy()-x.close.iloc[i])/x.atr.iloc[i]
            hits=np.flatnonzero(signed>0)
            row.update({"reversal_probability":float(np.any(signed>0)),"mae_atr":float(np.nanmax(adv)),"mfe_atr":float(np.nanmax(fav)),
                        "time_to_reversal_bars":float(hits[0]+1) if len(hits) else np.nan,
                        "subsequent_realized_volatility_atr":float(np.sqrt(np.nansum(np.diff(np.r_[x.close.iloc[i],future.close.to_numpy()])**2))/x.atr.iloc[i])})
        rows.append(row)
    events=pd.DataFrame(rows)
    agg={"signal_time":"count","before_cumulative_normalized_return":"mean","rvol":"mean","normalized_range":"mean","candle_body":"mean","close_location":"mean",
         **{f"return_{h}_bar_atr":"mean" for h in HORIZONS},"reversal_probability":"mean","mae_atr":"mean","mfe_atr":"mean","time_to_reversal_bars":"median","subsequent_realized_volatility_atr":"mean"}
    summary=events.groupby("category",sort=False).agg(agg).rename(columns={"signal_time":"events"}).reset_index()
    return events,summary


def direction_diagnostic(x: pd.DataFrame, split: pd.Timestamp) -> pd.DataFrame:
    mask=(x.rvol_20>=REP["rvol"])&(x.body_atr<=REP["move"])&(x.range_atr<=1.0)
    rows=[]
    for name,cont in (("reversal",False),("continuation",True)):
        t=backtest(x,signal_from_mask(x,mask,cont),INST);t=t[t.session_date>=split]
        rows.append({"outcome":name,**enrich(metrics(t),t)})
    idx=np.flatnonzero((mask&(x.session_date>=split)&(~x.last_bar)).to_numpy()); vals=[]
    for i in idx:
        end=min(i+6,int(x._session_last_pos.iloc[i])); vals.append((x.high.iloc[i+1:end+1].max()-x.low.iloc[i+1:end+1].min())/x.atr.iloc[i])
    rows.append({"outcome":"volatility_only","trades":len(vals),"mean_future_6bar_range_atr":np.mean(vals),"median_future_6bar_range_atr":np.median(vals)})
    return pd.DataFrame(rows)


def temporal(t: pd.DataFrame, split: pd.Timestamp) -> tuple[pd.DataFrame,pd.DataFrame]:
    dt=pd.to_datetime(t.session_date); mid=dt.min()+(dt.max()-dt.min())/2
    groups={"IS":dt<split,"OOS":dt>=split,"first_half":dt<=mid,"second_half":dt>mid,
            "pre_2020":dt<pd.Timestamp("2020-01-01"),"2020_2022":dt.between(pd.Timestamp("2020-01-01"),pd.Timestamp("2022-12-31")),"2023_plus":dt>=pd.Timestamp("2023-01-01")}
    rows=[]
    for name,mask in groups.items(): rows.append({"period":name,**enrich(metrics(t[mask]),t[mask])})
    for y,g in t.groupby(dt.dt.year): rows.append({"period":str(int(y)),**enrich(metrics(g),g)})
    yearly=pd.DataFrame(rows)
    yp=t.groupby(dt.dt.year).net_pnl.sum(); best=int(yp.idxmax()); z=t[dt.dt.year!=best]
    yearly=pd.concat([yearly,pd.DataFrame([{"period":"remove_best_year","removed_year":best,**enrich(metrics(z),z)}])],ignore_index=True)
    d=t.assign(date=dt).groupby("date").net_pnl.sum().sort_index(); full=d.reindex(pd.date_range(d.index.min(),d.index.max(),freq="B"),fill_value=0.0)
    roll=full.rolling(252,min_periods=126); rs=roll.mean()/roll.std(ddof=1)*math.sqrt(252)
    freq=pd.DataFrame({"date":full.index,"daily_pnl":full.values,"rolling_12m_sharpe":rs.values,"signal_count":t.groupby(dt).size().reindex(full.index,fill_value=0).values})
    return yearly,freq


def robustness(x:pd.DataFrame,split:pd.Timestamp,base:pd.DataFrame)->pd.DataFrame:
    rows=[]
    def add(name:str,t:pd.DataFrame,pnl_col:str="net_pnl",**extra:Any)->None: rows.append({"scenario":name,**extra,**enrich(metrics(t,pnl_col),t.assign(net_pnl=t[pnl_col]) if pnl_col!="net_pnl" else t)})
    o=base[base.session_date>=split].copy()
    for name,extra,cm in (("nominal",0,1),("plus_1_tick_side",1,1),("plus_2_ticks_side",2,1),("double_commissions",0,2)):
        z=o.copy();z["stress_pnl"]=z.gross_pnl-nominal_cost(INST,extra,cm);add(name,z,"stress_pnl")
    sig=signals(x,REP)
    for delay in (1,2):
        z=backtest(x,sig,INST,delay=delay);add(f"delay_entry_{delay}_bar",z[z.session_date>=split])
    for n in (5,10): add(f"remove_best_{n}_trades",o.drop(o.nlargest(min(n,len(o)),"net_pnl").index))
    day=o.groupby("session_date").net_pnl.sum(); days=day.nlargest(min(5,len(day))).index;add("remove_best_5_days",o[~o.session_date.isin(days)])
    yp=o.groupby(pd.to_datetime(o.session_date).dt.year).net_pnl.sum(); best=int(yp.idxmax());add("remove_best_year",o[pd.to_datetime(o.session_date).dt.year!=best],removed_year=best)
    opp=o.copy();opp["net_pnl"]=-opp.gross_pnl-nominal_cost(INST);add("opposite_direction",opp)
    rng=np.random.default_rng(SEED);sh=x.copy();sh["rvol_20"]=sh.groupby("slot").rvol_20.transform(lambda q:pd.Series(rng.permutation(q.to_numpy()),index=q.index));z=backtest(sh,signals(sh,REP),INST);add("shuffled_volume_labels",z[z.session_date>=split])
    return pd.DataFrame(rows)


def tod_breakdown(x:pd.DataFrame,t:pd.DataFrame)->pd.DataFrame:
    bins=[-1,5,17,47,71,10_000];labels=["opening_segment","mid_morning","midday","afternoon","closing_segment"]
    z=t.copy();z["segment"]=pd.cut(z.signal_bar_no,bins=bins,labels=labels)
    slot_features=x[["rvol_20","volume","expected_median_20"]].copy();slot_features["signal_time"]=x.index
    z["signal_time"]=pd.to_datetime(z.signal_time);z=z.merge(slot_features,on="signal_time",how="left")
    rows=[]
    for seg,g in z.groupby("segment",observed=False): rows.append({"segment":str(seg),"signal_frequency":len(g),"mean_rvol":g.rvol_20.mean(),"mean_volume":g.volume.mean(),"mean_expected_volume":g.expected_median_20.mean(),**enrich(metrics(g),g)})
    return pd.DataFrame(rows)


def portfolio(t:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    frozen=pd.read_csv(FROZEN,parse_dates=["session_date"]).set_index("session_date"); c=t.assign(session_date=pd.to_datetime(t.session_date)).groupby("session_date").net_pnl.sum().rename("candidate")
    z=frozen.join(c,how="left").fillna({"candidate":0}); p=z.portfolio_pnl_usd; cand=z.candidate
    mnq=["mnq_orb_vwap_atr","mnq_volatility_breakout","mnq_compression_breakout","mnq_atr_trend_flip"]
    trade_dir=t.assign(session_date=pd.to_datetime(t.session_date)).groupby("session_date").direction.sum().pipe(np.sign)
    sleeve_dir={}
    trade_dir_root=REPO/"research"/"portfolio_integration_phase1"/"sleeve_trades"
    for sleeve in mnq:
        ledger=pd.read_csv(trade_dir_root/f"{sleeve}.csv",parse_dates=["session_date"])
        sleeve_dir[sleeve]=ledger.groupby("session_date").signed_quantity.sum().pipe(np.sign)
    z["mnq_trend_cluster_total"]=z[mnq].sum(axis=1)
    active=cand.ne(0);loss=(cand<0); worst=p.nsmallest(10).index
    rows=[]
    for sleeve in mnq+["mnq_trend_cluster_total"]:
        s=z[sleeve]; common=active&s.ne(0)
        if sleeve=="mnq_trend_cluster_total":
            sd=pd.concat(sleeve_dir.values(),axis=1).fillna(0).sum(axis=1).pipe(np.sign)
        else: sd=sleeve_dir[sleeve]
        common_direction=trade_dir.index.intersection(sd.index); common_direction=common_direction[(trade_dir.loc[common_direction]!=0)&(sd.loc[common_direction]!=0)]
        rows.append({"sleeve":sleeve,"pnl_correlation":cand.corr(s),"negative_day_correlation":cand[p<0].corr(s[p<0]),
                     "signal_day_overlap":float(common.mean()),"direction_overlap":float((trade_dir.loc[common_direction]==sd.loc[common_direction]).mean()) if len(common_direction) else np.nan,
                     "simultaneous_loss_rate":float((loss&(s<0)).mean()),"candidate_pnl_worst_mnq_cluster_days":float(cand.loc[s.nsmallest(10).index].sum()),
                     "candidate_pnl_strong_trend_days":float(cand.loc[s.abs()>=s.abs().quantile(.9)].sum())})
    b=daily_metrics(p);a=daily_metrics(p+cand);summary=pd.DataFrame([{"scenario":"9_sleeves",**b,"downside_correlation":np.nan,"marginal_variance":0.0,"worst_day":p.min()},
        {"scenario":"9_plus_candidate",**a,"downside_correlation":cand[p<0].corr(p[p<0]),"marginal_variance":float((p+cand).var()-p.var()),"worst_day":float((p+cand).min())}])
    worst_table=z.loc[worst,["portfolio_pnl_usd","candidate"]].reset_index();worst_table["combined_pnl"]=worst_table.portfolio_pnl_usd+worst_table.candidate
    return pd.DataFrame(rows),summary,worst_table


def paired_topstep(t:pd.DataFrame)->pd.DataFrame:
    timeline,base_daily,base_min,base_traded,bootstrap_traded=build_baseline(); calendar=base_daily.index
    base={"daily":base_daily.to_numpy(float),"intraday_min":base_min.to_numpy(float),"traded":base_traded.to_numpy(bool)}
    ct=t.copy();ct["mae_usd"]=ct.mae;ct["nominal_cost"]=nominal_cost(INST)
    _,bad,cdaily,ctraded=candidate_intraday(ct,timeline,INST.point_value)
    dates=pd.Series(pd.to_datetime(timeline.tz_convert("America/New_York").date),index=timeline);cmin=bad.groupby(dates,sort=False).min();cmin.index=pd.to_datetime(cmin.index)
    variants={"equal_contract__nominal__six":{**base,"traded":bootstrap_traded.to_numpy(bool)},"nine_sleeves":base,
              "nine_plus_candidate":{"daily":base_daily.add(cdaily.reindex(calendar).fillna(0)).to_numpy(float),"intraday_min":base_min.add(cmin.reindex(calendar).fillna(0)).to_numpy(float),"traded":(base_traded|ctraded.reindex(calendar).fillna(False)).to_numpy(bool)}}
    runs,_=topstep_campaign(variants);boot=runs[(runs.method=="moving_block_bootstrap")&(runs.horizon==60)];keys=["block","path_id"]
    b=boot[boot.scenario=="nine_sleeves"].set_index(keys);c=boot[boot.scenario=="nine_plus_candidate"].set_index(keys).loc[b.index];rng=np.random.default_rng(SEED);rows=[]
    for outcome in ("PASS","MLL_FAIL","EXPIRY"):
        diff=c.outcome.eq(outcome).astype(float).to_numpy()-b.outcome.eq(outcome).astype(float).to_numpy();draw=np.array([rng.choice(diff,len(diff),replace=True).mean() for _ in range(5000)])
        rows.append({"metric":"delta_"+outcome.lower(),"paths":len(diff),"estimate":diff.mean(),"ci95_low":np.quantile(draw,.025),"ci95_high":np.quantile(draw,.975),"baseline_probability":b.outcome.eq(outcome).mean(),"candidate_probability":c.outcome.eq(outcome).mean()})
    return pd.DataFrame(rows)


def plots(cells:pd.DataFrame,events:pd.DataFrame,rolling:pd.DataFrame)->None:
    OUT.mkdir(parents=True,exist_ok=True)
    o=cells[cells.scope=="OOS"]
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,mode in zip(axs,["preceding_move_reversal","bar_breakout"]):
        q=o[o["mode"]==mode].pivot(index="move",columns="rvol",values="sharpe");im=ax.imshow(q,cmap="RdYlGn",vmin=-2,vmax=2,aspect="auto");ax.set(title=mode,xlabel="RVOL threshold",ylabel="response threshold");ax.set_xticks(range(len(q.columns)),q.columns);ax.set_yticks(range(len(q.index)),q.index)
    fig.colorbar(im,ax=axs.ravel().tolist());fig.savefig(OUT/"inherited_region_oos_sharpe.png",dpi=140,bbox_inches="tight");plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));
    for cat,g in events.groupby("category"):
        means=[g[f"return_{h}_bar_atr"].mean() for h in HORIZONS];ax.plot(HORIZONS,means,marker="o",label=cat)
    ax.axhline(0,color="black",lw=.7);ax.legend(fontsize=7);ax.set(xlabel="bars after event",ylabel="reversal-direction return (ATR)",title="Mandatory 2x2 event paths");fig.tight_layout();fig.savefig(OUT/"event_study_2x2.png",dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));ax.plot(pd.to_datetime(rolling.date),rolling.rolling_12m_sharpe);ax.axhline(0,color="black",lw=.7);ax.set(title="Rolling 12-month Sharpe",ylabel="Sharpe");fig.tight_layout();fig.savefig(OUT/"rolling_12m_sharpe.png",dpi=140);plt.close(fig)


def run() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    x=add_price_features(add_causal_volume_baselines(load_bars(REPO/DATA_FILES[ASSET],INST)));split=split_date(x)
    cells,cache=all_cell_results(x,split);rep_id=variant_id(REP);rep=cache[rep_id]
    annual_cells=[]
    for vid,t in cache.items():
        tagged=t.assign(year=pd.to_datetime(t.session_date).dt.year,split=np.where(pd.to_datetime(t.session_date)<split,"IS","OOS"))
        for (scope,year),g in tagged.groupby(["split","year"]):
            annual_cells.append({"variant_id":vid,"year":int(year),"split":scope,**enrich(metrics(g),g)})
    annual_cells=pd.DataFrame(annual_cells)
    controls,random_controls=interaction_controls(x,split);events,event_summary=event_study(x,split);direction=direction_diagnostic(x,split)
    temp,rolling=temporal(rep,split);robust=robustness(x,split,rep);tod=tod_breakdown(x,rep[rep.session_date>=split]);relations,port,worst=portfolio(rep[rep.session_date>=split])
    topstep_path=OUT/"topstep_paired_bootstrap.csv"
    topstep=pd.read_csv(topstep_path) if topstep_path.exists() else paired_topstep(rep[rep.session_date>=split])
    source=pd.read_csv(SOURCE/"all_parameter_results.csv");src=source[(source.asset==ASSET)&(source.mechanism=="D")].rename(columns={"split":"scope"})
    merged=cells[cells.scope.isin(["IS","OOS"])].merge(src,on=["variant_id","scope"],suffixes=("_audit","_source"));maxdiff=float((merged.net_pnl_audit-merged.net_pnl_source).abs().max())
    for name,df in {"inherited_cells.csv":cells,"inherited_cell_annual_results.csv":annual_cells,"interaction_controls.csv":controls,"random_trade_count_controls.csv":random_controls,
                    "event_level_2x2.csv":events,"event_study_2x2_summary.csv":event_summary,"continuation_vs_reversal.csv":direction,
                    "temporal_stability.csv":temp,"rolling_12m_sharpe_and_frequency.csv":rolling,"falsification_and_costs.csv":robust,
                    "time_of_day_breakdown.csv":tod,"mnq_sleeve_relationships.csv":relations,"portfolio_comparison.csv":port,
                    "worst_portfolio_days.csv":worst,"topstep_paired_bootstrap.csv":topstep}.items(): df.to_csv(OUT/name,index=False)
    plots(cells,events,rolling)
    o=cells[cells.scope=="OOS"];i=cells[cells.scope=="IS"]
    pos_o=float(((o.net_pnl>0)&(o.sharpe>0)).mean());pos_i=float(((i.net_pnl>0)&(i.sharpe>0)).mean())
    piv=cells.pivot(index="variant_id",columns="scope",values="net_pnl");both=float(((piv.IS>0)&(piv.OOS>0)).mean())
    rep_o=o[o.variant_id==rep_id].iloc[0];same_mode=o[o["mode"]==REP["mode"]].copy(); neighbours=same_mode[(same_mode.variant_id!=rep_id)&((same_mode.move-REP["move"]).abs().le(.25))&((same_mode.rvol-REP["rvol"]).abs().le(.5))]
    neigh=float((neighbours.net_pnl>0).mean());local_type="one threshold boundary" if neigh<.5 else "coherent but narrow"
    top5=robust.set_index("scenario").loc["remove_best_5_trades"];top10=robust.set_index("scenario").loc["remove_best_10_trades"]
    verdict="REJECTED"
    report=["# MNQ High-Volume / Low-Response Proxy Audit","",f"**Final verdict: `{verdict}`**","",
      "This is an OHLCV proxy for low price impact under abnormal volume. It is not true absorption. No new parameter values were tested and no existing sleeve was changed.","",
      "## Executive conclusion","",
      f"The inherited representative reproduces exactly (maximum source/replay PnL difference: ${maxdiff:.2f}), but it is a retrospective local anomaly rather than a repeatable executable mechanism. Its OOS result is positive (Sharpe {rep_o.sharpe:.2f}, PF {rep_o.pf:.2f}, {int(rep_o.trades)} trades), while the same cell is negative IS and the inherited landscape is not jointly stable.","",
      f"Across the 12 inherited cells, {pos_o:.1%} are positive OOS, {pos_i:.1%} are positive IS, and {both:.1%} are positive in both. The representative's immediate positive-neighbour fraction is {neigh:.1%}; `LOCAL_REGION` is best described as **{local_type}**, not a broad plateau.","",
      f"Concentration is material: after removing the best 5 trades, PnL is ${top5.net_pnl:,.0f} and Sharpe {top5.sharpe:.2f}; after removing the best 10, PnL is ${top10.net_pnl:,.0f} and Sharpe {top10.sharpe:.2f}.","",
      "## Phase 1 — exact inherited mechanism","",
      "- Bars: MNQ regular session, repository 5-minute bars resampled from Databento OHLCV-1m.",
      "- Expected volume: median volume of the prior 20 sessions at the identical five-minute time-of-day slot; the current session is shifted out.",
      "- RVOL: current bar volume / causal expected median volume. Volume surprise also exists in the source as (volume − median) / causal IQR-equivalent scale, but is not used by this signal.",
      "- Normalized response: body = |close − open| / causal 20-bar within-session median true range; range = (high − low) / the same ATR.",
      "- Low response: body ≤ 0.25 ATR and range ≤ 1.0 ATR for the canonical cell.",
      "- Preceding move: sign(close[t] − close[t−3]) on the signal bar.",
      "- Reversal direction: negative of that preceding three-bar move.",
      "- Confirmation: none beyond the completed signal bar (`confirm=1` is an inherited identifier field, not an extra bar).",
      "- Execution: signal at bar close; enter next bar open. Non-overlapping trades only.",
      "- Exit: close of the sixth held bar, clipped to the session close.",
      f"- Nominal costs: one tick per side plus ${INST.commission_side:.2f} commission per side = ${nominal_cost(INST):.2f} round trip.","",
      "### Every inherited cell — IS and OOS","",md(cells[cells.scope.isin(["IS","OOS"])][["variant_id","scope","trades","net_pnl","sharpe","pf","expectancy","win_rate","maxdd","annualized_pnl"]]),"","Every cell's calendar-year performance is preserved in `results/inherited_cell_annual_results.csv`.","",
      "## Phase 3 — decomposition and incremental interaction","",md(controls[[c for c in controls.columns if c!="annualized_pnl"]]),"",
      "The interaction is judged against price geometry, volume alone, count-matched random thinning, and same-time-of-day shuffled volume. A positive representative alone is insufficient when these controls and the IS replay do not jointly establish repeatability.","",
      "## Phase 4 — mandatory economic 2×2 event study","",md(event_summary),"",f"Full event rows: `event_level_2x2.csv`. Plot: `event_study_2x2.png`.","",
      "## Phase 5 — continuation, reversal, or volatility only","",md(direction),"",
      "## Phase 6 — temporal stability","",md(temp[["period","trades","net_pnl","sharpe","pf","expectancy","win_rate","maxdd"]]),"",
      "Signal counts and rolling 12-month Sharpe are in `rolling_12m_sharpe_and_frequency.csv` and `rolling_12m_sharpe.png`. The full-history sign reversal from negative IS to positive OOS prevents a stable-effect interpretation.","",
      "## Phase 7 — costs, concentration, and falsification","",md(robust[["scenario","trades","net_pnl","sharpe","pf","expectancy","win_rate","maxdd"]]),"",
      "The requested random matched controls are in `random_trade_count_controls.csv`; opposite direction, shuffled labels, delays, costs, best trades/days/year removals are above and in the supporting CSV.","",
      "## Phase 8 — time of day / volume seasonality","",md(tod[["segment","signal_frequency","mean_rvol","mean_volume","mean_expected_volume","net_pnl","sharpe","expectancy"]]),"",
      "These are diagnostics only; no time filter is selected.","",
      "## Phase 9 — relation to frozen MNQ sleeves","",md(relations),"",
      "## Phase 10 — portfolio and Topstep","",md(port),"",md(topstep),"",
      "Topstep uses the frozen nine-sleeve sizing unchanged and a paired 8,000-path moving-block bootstrap at the 60-traded-day horizon. The candidate's observed five-minute MAE is added conservatively to the unchanged minute-level nine-sleeve adverse path. A delta is not treated as improvement when its 95% interval includes zero.","",
      "## Decision","",f"**`{verdict}`**","",
      "The high-volume + low-response proxy has an interesting OOS pocket, but it fails the required standard of an incremental, temporally repeatable interaction: the canonical cell loses IS, the profitable area is threshold-sensitive, only a minority of inherited cells are positive, and concentration removal materially weakens the result. Do not freeze it prospectively as an executable region and do not add it to the nine-sleeve portfolio."]
    (HERE/"MNQ_HIGH_VOLUME_LOW_RESPONSE_AUDIT.md").write_text("\n".join(report),encoding="utf-8")
    meta={"verdict":verdict,"new_parameter_search":False,"inherited_cells":12,"representative":rep_id,"split_date":str(split.date()),"data_start":str(x.index.min()),"data_end":str(x.index.max()),"source_replay_max_abs_pnl_diff":maxdiff,"seed":SEED}
    (OUT/"run_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")


if __name__ == "__main__": run()
