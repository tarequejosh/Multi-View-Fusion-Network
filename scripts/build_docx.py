#!/usr/bin/env python3
"""
Build paper/manuscript.docx from paper/manuscript.tex with pandoc, so the Word
version is never edited by hand. IEEEtran-only commands that pandoc cannot
handle (drop cap, running heads) are removed in a temporary copy first.

Usage: python scripts/build_docx.py   (requires pandoc on PATH)
"""

import subprocess
import tempfile
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent / "paper"


def main():
    s = (PAPER / "manuscript.tex").read_text(encoding="utf-8")
    s = s.replace(r"\IEEEPARstart{B}{rain}", "Brain")
    start = s.find(r"\markboth")
    if start >= 0:
        end = s.index("\n", s.index("\n", start) + 1) + 1
        s = s[:start] + s[end:]
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "manuscript.tex"
        src.write_text(s, encoding="utf-8")
        subprocess.run(["pandoc", str(src), "--from=latex", "--citeproc", "--bibliography=references.bib",
                        "--resource-path=.;../figures/journal;../figures/external", "-o", "manuscript.docx"],
                       cwd=PAPER, check=True)
    print("written", PAPER / "manuscript.docx")


if __name__ == "__main__":
    main()
