"""The README's front door only points at things that exist."""

from __future__ import annotations

import re

from tests.support import ROOT

README = (ROOT / "README.md").read_text(encoding="utf-8")
DECISIONS = (ROOT / "DECISIONS.md").read_text(encoding="utf-8")


def test_every_decision_the_readme_cites_exists() -> None:
    cited = {int(n) for n in re.findall(r"ADR-(\d{3})", README)}
    present = {int(n) for n in re.findall(r"^## ADR-(\d{3}):", DECISIONS, flags=re.M)}
    assert cited and cited <= present, sorted(cited - present)


def test_every_range_of_decisions_the_readme_cites_is_complete() -> None:
    present = {int(n) for n in re.findall(r"^## ADR-(\d{3}):", DECISIONS, flags=re.M)}
    for low, high in re.findall(r"ADR-(\d{3}) to ADR-(\d{3})", README):
        assert set(range(int(low), int(high) + 1)) <= present, (low, high)


def test_the_files_the_front_door_names_exist() -> None:
    top = README.split("## Quickstart")[0]
    for path in re.findall(r"`((?:docs|evals|src|scripts|deploy)/[\w./-]+|[A-Z]+\.md)`", top):
        assert (ROOT / path).exists(), path


def test_the_status_line_is_not_stale() -> None:
    assert "Week 1 in progress" not in README


def test_the_front_door_states_its_limits() -> None:
    assert "## Known limits" in README and "## What is proven, and how" in README


WRITEUP = (ROOT / "docs" / "WRITEUP.md").read_text(encoding="utf-8")


def test_every_decision_the_writeup_cites_exists() -> None:
    cited = {int(n) for n in re.findall(r"ADR-(\d{3})", WRITEUP)}
    present = {int(n) for n in re.findall(r"^## ADR-(\d{3}):", DECISIONS, flags=re.M)}
    assert cited and cited <= present, sorted(cited - present)


def test_the_writeup_says_what_it_did_not_measure() -> None:
    assert "## 4. What I did not do, and what is not known" in WRITEUP
    assert "One run per case" in WRITEUP and "synthetic" in WRITEUP


def test_the_commands_the_writeup_gives_exist() -> None:
    for module in set(re.findall(r"python -m (kartly[\w.]*)", WRITEUP)):
        path = ROOT / "src" / module.replace(".", "/")
        assert path.with_suffix(".py").exists() or (path / "__main__.py").exists(), module
