"""Regenerate one signal-sheet HTML without touching the other sheets."""

from __future__ import annotations

import argparse
from pathlib import Path

from export_signal_sheets_pdf import (
    SOURCE_DIR,
    interactive_sheet_figures,
    markdown_to_body,
    page,
    title_from_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True, choices=range(3, 10))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = next(path for path in SOURCE_DIR.glob(f"{args.index:02d}_*.md"))
    text = source.read_text(encoding="utf-8")
    title = title_from_markdown(text)
    body = markdown_to_body(text, source, interactive_figures=interactive_sheet_figures(args.index))
    edition = "1 NOVEMBRE 2026"
    decorated = f'<div class="document-meta">FICHE SIGNAL {args.index:02d}/09 · DOCUMENT CLIENT · ÉDITION {edition}</div>' + body
    destination = args.output_dir / f"{source.stem}.html"
    destination.write_text(page(title, decorated, "mermaid.min.js", "plotly.min.js"), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
