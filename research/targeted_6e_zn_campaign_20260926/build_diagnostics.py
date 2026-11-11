"""Build reporting-only diagnostics from the completed locked campaign."""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main() -> None:
    cfg = json.loads((HERE / "campaign_config.json").read_text())
    trades = pd.read_csv(HERE / "trades.csv", parse_dates=["entry_ts", "exit_ts"])
    start, end = cfg["periods"]["full"]
    index = pd.bdate_range(start, end)
    curves = {}
    for (framework, variant), frame in trades.groupby(["framework", "variant"]):
        pnl = frame.groupby(pd.to_datetime(frame.trade_date)).net_pnl.sum()
        curves[f"{framework}__{variant}"] = pnl.reindex(index, fill_value=0.0)
    pd.DataFrame(curves, index=index).rename_axis("date").to_csv(HERE / "daily_pnl.csv")

    audits = []
    for asset, spec in cfg["instruments"].items():
        data = pd.read_parquet(ROOT / spec["data_path"], columns=["close"])
        idx = pd.DatetimeIndex(data.index)
        audits.append({
            "asset": asset, "rows": len(data), "first_bar": str(idx.min()),
            "last_bar": str(idx.max()), "index_timezone": str(idx.tz),
            "duplicate_timestamps": int(idx.duplicated().sum()),
            "monotonic": bool(idx.is_monotonic_increasing),
        })
    pd.DataFrame(audits).to_csv(HERE / "data_audit.csv", index=False)


if __name__ == "__main__":
    main()
