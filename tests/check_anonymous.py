"""Anonymous release checklist for SES-3D core code.

Run before uploading a supplementary zip / anonymous repository.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Patterns that must not appear in the release tree.
FORBIDDEN = [
    re.compile(r"[A-Za-z]:\\\\Users\\\\", re.I),
    re.compile(r"/media/"),
    re.compile(r"/home/"),
    re.compile(r"/mnt/"),
    re.compile(r"\bZZL\b"),
    re.compile(r"\bZzil\b"),
    re.compile(r"XiaomiMiMo", re.I),
    re.compile(r"codex\\\\2026", re.I),
    re.compile(r"@[a-z0-9.-]+\.(com|org|edu|cn)", re.I),
    re.compile(r"github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"),
    re.compile(r"anonymous\s+review\s+link\s+placeholder", re.I),
]

ALLOWED_SUFFIXES = {".py", ".md", ".txt", ".toml", ".cfg", ".ini", ""}


def main() -> int:
    bad = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "check_anonymous.py":
            continue
        if path.suffix.lower() not in {".py", ".md", ".txt", ".toml"} and path.name not in {
            "LICENSE",
            "requirements.txt",
        }:
            # skip caches / zips produced next to the tree
            if path.suffix.lower() in {".zip", ".pyc"} or "__pycache__" in path.parts:
                continue
        if path.suffix.lower() == ".zip" or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pat in FORBIDDEN:
            m = pat.search(text)
            if m:
                rel = path.relative_to(ROOT)
                print(f"FORBIDDEN {rel}: {pat.pattern} -> {m.group(0)!r}")
                bad += 1
    if bad:
        print(f"\n{bad} issue(s). Remove identifying strings before release.")
        return 1
    print("Anonymous scan clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
