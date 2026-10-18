"""Lighter stays small, and the README says how small."""

import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
LIMIT = 1400


def test_source_stays_under_the_limit_and_matches_the_readme():
    lines = sum(len(path.read_text().splitlines()) for path in (ROOT / "src" / "lighter").rglob("*.py"))
    assert lines <= LIMIT, f"src/lighter has {lines} lines, more than {LIMIT}. Delete something first."
    stated = re.search(r"~([\d,]+) lines of code", (ROOT / "README.md").read_text())
    assert stated, "README.md should state the size as '~N lines of code'."
    claimed = int(stated.group(1).replace(",", ""))
    assert abs(claimed - lines) <= 0.1 * lines, f"README.md says ~{claimed} lines of code; src/lighter has {lines}."
