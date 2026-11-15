"""Single entry point: ``python -m audit_20261115.run_all [--no-engine]``.

Phases 0-3, 4a (hash check only), 5-10 and the HTML export.  The preregistration is written once and never
rewritten: only its sha256 (and that of outputs/04_criteria.csv) is checked against preregistration.sha256.
The one-shot reads are NOT part of this command: phase4b_s01_oneshot.py (done once, refuses a rerun) and
forward_check.py (prospective reads, once per horizon).  Phase 6 reads the stored 4b result.
Engine re-executions of phase 0 are cached in audit_20261115/cache/ (delete it to force a ~35 min re-run).
Decisions of 2026-11-15: S01 = 18-model ATR ensemble of signal sheet 01; Sharpe in USD for every sleeve.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys

from .common import AUDIT, OUT, ROOT, SEED, sha256, write_json


def versions() -> dict:
    import numpy, pandas, pyarrow, scipy  # noqa: E401
    import subprocess
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"python": sys.version, "platform": platform.platform(), "numpy": numpy.__version__,
            "pandas": pandas.__version__, "pyarrow": pyarrow.__version__, "scipy": scipy.__version__,
            "git_head": head, "seed": SEED}


def check_prereg() -> bool:
    lines = (AUDIT / "preregistration.sha256").read_text(encoding="utf-8").split("\n")
    expected = dict(reversed(line.split("  ", 1)) for line in lines if line.strip())
    ok = (sha256(AUDIT / "preregistration.md") == expected["preregistration.md"]
          and sha256(OUT / "04_criteria.csv") == expected["outputs/04_criteria.csv"])
    if not ok:
        raise SystemExit("preregistration.md or 04_criteria.csv changed since it was hashed")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-engine", action="store_true", help="skip the raw-bar re-execution (level B)")
    args = ap.parse_args()
    write_json(OUT / "run_metadata.json", versions())
    from . import phase0_reproduction, phase1_calendar, phase2_significance, phase3_inventory, render_phase0
    r = phase0_reproduction.main(engine=not args.no_engine)
    render_phase0.main()
    print("phase 0:", json.dumps({k: r[k] for k in ("all_ok", "n_checks", "n_fail")}),
          "(2 known gaps, decided on 2026-11-15)")
    phase1_calendar.main()
    phase2_significance.main()
    phase3_inventory.main()
    if (AUDIT / "preregistration.md").exists():
        print("phase 4a: preregistration exists, hash verified:", check_prereg())
    else:
        from . import phase4a_prereg
        print("phase 4a:", phase4a_prereg.main())
    from . import (phase5_stop_rules, phase6_status, phase7_vote, phase8_regimes, phase9_topstep, phase10_cat_stop,
                   render_html)
    phase5_stop_rules.main()
    if (OUT / "04b_result.json").exists():
        phase6_status.main()
    phase7_vote.main()
    phase8_regimes.main()
    phase9_topstep.main()
    phase10_cat_stop.main()
    render_html.main()
    print("phases 5-10 done")


if __name__ == "__main__":
    main()
