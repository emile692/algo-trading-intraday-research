"""HTML export of the audit (RAPPORT.md + one page per phase), with the repository's signal-sheet stylesheet.

Uses scripts/export_signal_sheets_pdf.py (CSS and page template) and the `markdown` package; no network access.
Links to .md files are rewritten to the matching .html pages.  Output: audit_20261115/html/.
"""

from __future__ import annotations

import html
import re

import markdown

from .common import AUDIT

PAGES = ["RAPPORT.md", "00_reproduction.md", "01_calendrier.md", "02_significativite.md", "03_inventaire_holdout.md",
         "preregistration.md", "04_holdout.md", "05_regles_arret.md", "06_coherence_statuts.md", "07_vote.md",
         "08_regimes.md", "09_topstep.md", "10_stop_catastrophe.md"]


def main() -> list[str]:
    import export_signal_sheets_pdf as ex
    out_dir = AUDIT / "html"
    out_dir.mkdir(exist_ok=True)
    written = []
    for name in PAGES:
        src = AUDIT / name
        if not src.exists():
            continue
        text = src.read_text(encoding="utf-8")
        text = re.sub(r"\]\((?!https?:)(?!\.\./)([^)#]+?)\.md(#[^)]*)?\)", lambda m: f"]({m.group(1)}.html{m.group(2) or ''})", text)
        text = re.sub(r"\]\(\.\./", "](../../", text)  # links leaving audit_20261115/ (html/ is one level deeper)
        text = re.sub(r"\]\((?!https?:)(?!\.\./)((?:outputs|cache)/[^)]+|[a-z0-9_]+\.py|05_suivi_hebdo_gabarit\.csv)\)",
                      lambda m: f"](../{m.group(1)})", text)
        body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists", "toc"], output_format="html5")
        title = re.search(r"(?m)^#\s+(.+)$", text).group(1).strip()
        nav = '<div class="document-meta">AUDIT 2026-11-15 · <a href="RAPPORT.html">rapport</a></div>'
        page = ex.page(title, nav + body, "", "").replace('<script src=""></script>', "")
        dest = out_dir / name.replace(".md", ".html")
        dest.write_text(page, encoding="utf-8")
        written.append(dest.name)
    return written


if __name__ == "__main__":
    print(main())
