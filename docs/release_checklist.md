# Release checklist

Zenodo archives every **published GitHub release** of this repository and mints a DOI for it, using the
metadata in `.zenodo.json` (not `CITATION.cff`). Tags without a published release don't trigger it. PyPI never
allows a version to be replaced, so every upload needs a new version number.

## One-time Zenodo setup (maintainer)
1. <https://zenodo.org> → **Log in with GitHub**.
2. **Account menu → GitHub** → **Sync now** if `Gideono29/lablite-cvd` isn't listed → switch it **On**.
3. Check GitHub **Settings → Webhooks** lists a `zenodo.org` webhook.

## Before a release
- [ ] Version identical in `pyproject.toml`, `src/lablite_cvd/__init__.py`, `CITATION.cff`
      (`tests/test_release_metadata.py` checks).
- [ ] `CITATION.cff` `date-released:` and a dated `CHANGELOG.md` entry.
- [ ] Full rerun from public data if the model changed: `lablite-cvd download`, `cohort`, `fit`, `bootstrap`,
      `python scripts/subgroup_performance.py`; copy `outputs/fit/lablite_params.json` to
      `src/lablite_cvd/data/` (`tests/test_packaged_model.py` checks they match); update the model card.
- [ ] `pytest -q` passes; GitHub Actions green on `main`.
- [ ] Build and check: `python -m build`, `python -m twine check dist/*`; install the wheel in a clean venv.

## Upload to PyPI (maintainer's token; PowerShell)
Paste the token only at the `Paste token:` prompt; the username must stay `__token__`.
```powershell
$s = Read-Host "Paste token" -AsSecureString; $env:TWINE_USERNAME = "__token__"; $env:TWINE_PASSWORD = [Net.NetworkCredential]::new("", $s).Password; .venv\Scripts\python -m twine upload dist/*; Remove-Item Env:TWINE_PASSWORD
```
Add `--repository testpypi` to try TestPyPI first.

## GitHub release → Zenodo DOI
1. GitHub → **Releases → Draft a new release**, tag `vX.Y.Z` on `main`, title `LabLite-CVD vX.Y.Z`,
   description = the `CHANGELOG.md` entry. **Publish release.**
2. Zenodo shows the record within minutes, with a **version DOI** (this release) and a **concept DOI** (all
   versions; cite it in the CV and README).

## After release
- [ ] Add the concept DOI to `CITATION.cff` (`doi:` and `identifiers:`) and a DOI badge to `README.md`.
- [ ] Optionally deposit `data/processed/cohort.csv.gz` and `outputs/fit/oof_predictions.csv.gz` on Zenodo
      as a dataset (excluded from git).
