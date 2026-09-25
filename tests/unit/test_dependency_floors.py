"""Dependency-floor guards for requirements.txt / pyproject.toml.

The repo states its install floors in two places that can silently drift
apart. These tests pin the invariant both ways:

1. every package present in BOTH files declares the SAME floor, and
2. the hard floors imposed by import-time / call-surface facts never regress:
   - torch >= 2.4.0: DCP ``async_save`` exists from 2.3 but the ``no_dist``
     kwarg our calls pass only exists from 2.4 (verified per-release-tag).
   - peft >= 0.19.0: ``peft.tuners.lora.loraga`` and ``LoraGAConfig`` first
     ship in 0.19.0 — both are imported at module load by sft_trainer.py.
   - trl >= 1.4.0: DPOConfig loss_type list, processing_class, GRPOConfig.
   - transformers >= 5.0: warmup_steps float-ratio semantics under test.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS = REPO_ROOT / "requirements.txt"
PYPROJECT = REPO_ROOT / "pyproject.toml"

_HARD_FLOORS: dict[str, str] = {
    "torch": "2.4.0",
    "peft": "0.19.0",
    "trl": "1.4.0",
    "transformers": "5.0",
}

_REQ_LINE = re.compile(r"^([A-Za-z0-9_.-]+)\s*>=\s*(\d[\w.]*)$")


def _parse_requirements() -> dict[str, str]:
    floors: dict[str, str] = {}
    for raw in REQUIREMENTS.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _REQ_LINE.match(line.split(";")[0].strip())
        assert match, f"unparsed requirements.txt line: {raw!r}"
        floors[match.group(1).lower()] = match.group(2)
    return floors


def _parse_pyproject() -> dict[str, str]:
    text = PYPROJECT.read_text()
    block = re.search(r"^dependencies\s*=\s*\[\n(.*?)^\]", text, re.S | re.M)
    assert block, "dependencies list not found in pyproject.toml"
    floors: dict[str, str] = {}
    for entry in re.findall(r'"([^"]+)"', block.group(1)):
        match = _REQ_LINE.match(entry.split(";")[0].strip())
        assert match, f"unparsed pyproject dependency: {entry!r}"
        floors[match.group(1).lower()] = match.group(2)
    return floors


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split(".") if part.isdigit())


def test_requirements_and_pyproject_floors_agree() -> None:
    req = _parse_requirements()
    proj = _parse_pyproject()
    shared = sorted(set(req) & set(proj))
    assert shared, "parser drift: no shared packages found"
    for name in shared:
        assert req[name] == proj[name], (
            f"{name}: requirements.txt floors {req[name]} but pyproject.toml "
            f"floors {proj[name]} — keep the two in sync"
        )


def test_hard_floors_are_not_lowered() -> None:
    for name, floor in _HARD_FLOORS.items():
        for source, floors in (
            ("requirements.txt", _parse_requirements()),
            ("pyproject.toml", _parse_pyproject()),
        ):
            assert name in floors, f"{name} missing from {source}"
            assert _version_tuple(floors[name]) >= _version_tuple(floor), (
                f"{name}>={floors[name]} in {source} is below the verified minimum {floor}"
            )
