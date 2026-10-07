import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import lablite_cvd

ROOT = Path(__file__).resolve().parents[1]


def test_versions_agree():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    version = pyproject["project"]["version"]
    assert lablite_cvd.__version__ == version
    assert f"version: {version}" in cff


def test_research_use_notice_in_readme_and_card():
    for name in ("README.md", "docs/model_card.md"):
        assert "research use only" in (ROOT / name).read_text(encoding="utf-8").lower(), name


def test_zenodo_metadata_is_valid():
    import json

    meta = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    assert meta["upload_type"] == "software" and meta["license"] == "MIT"
    assert meta["creators"][0]["orcid"] == "0009-0000-0540-7449"
