"""Download NHANES XPT component files and the 2019 public-use linked mortality files.

Ported from EquiCVD Bench v1.0.0.
"""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from lablite_cvd.pipeline.config import CYCLES, MORT_URL, NHANES_URLS, cycle_files

UA = {"User-Agent": "LabLite-CVD (research; python-requests)"}


def _looks_valid(path: Path, kind: str) -> bool:
    if not path.exists() or path.stat().st_size < 200:
        return False
    head = path.read_bytes()[:80]
    if kind == "xpt":
        return head.startswith(b"HEADER RECORD")
    return not head.lstrip().lower().startswith((b"<!doctype", b"<html"))


def _fetch(urls, dest: Path, kind: str, retries: int = 3) -> dict:
    if _looks_valid(dest, kind):
        return {"file": dest.name, "status": "cached", "url": None}
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for url in urls:
        for attempt in range(retries):
            try:
                r = requests.get(url, headers=UA, timeout=120)
                if r.status_code == 404:
                    last = f"404 {url}"
                    break
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                tmp.write_bytes(r.content)
                if not _looks_valid(tmp, kind):
                    tmp.unlink()
                    last = f"invalid content from {url}"
                    break
                tmp.replace(dest)
                return {"file": dest.name, "status": "downloaded", "url": url}
            except requests.RequestException as e:
                last = f"{type(e).__name__}: {e}"
                time.sleep(2 ** attempt)
    return {"file": dest.name, "status": "failed", "error": last}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_all(data_dir: Path, jobs: int = 4) -> list:
    data_dir = Path(data_dir)
    raw = data_dir / "raw"
    tasks = []
    for label, year, suffix in CYCLES:
        for stem in cycle_files(year, suffix):
            urls = [u.format(year=year, label=label, stem=stem) for u in NHANES_URLS]
            tasks.append((urls, raw / "nhanes" / label / f"{stem}.xpt", "xpt"))
        tasks.append(([MORT_URL.format(y0=year, y1=year + 1)],
                      raw / "mortality" / f"NHANES_{year}_{year + 1}_MORT_2019_PUBLIC.dat", "dat"))

    results = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_fetch, *t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            dest = futs[fut][1]
            res["path"] = dest.relative_to(data_dir).as_posix()
            if res["status"] != "failed":
                res["bytes"] = dest.stat().st_size
                res["sha256"] = _sha256(dest)
            results.append(res)
            print(f"[{i:3d}/{len(tasks)}] {res['status']:10s} {res['path']}", flush=True)

    results.sort(key=lambda r: r["path"])
    manifest = raw / "manifest.json"
    # Provenance only (stable across re-runs): keep the source URL recorded when a cached file was first fetched
    previous = {}
    if manifest.exists():
        previous = {r["path"]: r for r in json.loads(manifest.read_text())}
    entries = []
    for r in results:
        if r["status"] == "failed":
            entries.append({k: r.get(k) for k in ("file", "path", "status", "error")})
            continue
        url = r["url"] or previous.get(r["path"], {}).get("url")
        entries.append({"file": r["file"], "path": r["path"], "url": url, "bytes": r["bytes"], "sha256": r["sha256"]})
    manifest.write_text(json.dumps(entries, indent=1))
    failed = [r for r in results if r["status"] == "failed"]
    print(f"\n{len(results) - len(failed)}/{len(results)} files present; manifest -> {manifest}")
    for r in failed:
        print(f"  FAILED {r['path']}: {r.get('error')}")
    return results
