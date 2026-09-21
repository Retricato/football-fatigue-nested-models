#!/usr/bin/env python3
"""
04_fetch_soccermon.py — download, verify, unpack and inspect SoccerMon.

SoccerMon: Midoglu et al., Zenodo DOI 10.5281/zenodo.10033832, CC BY 4.0.
You MUST cite the dataset in your thesis; the licence requires attribution.

Typical Day-2 usage
-------------------
    python 04_fetch_soccermon.py --subjective          # 705 kB, do this first
    python 04_fetch_soccermon.py --inspect             # print real column names
    python 04_fetch_soccermon.py --inspect --save-report

Track B (Day 18 gate ONLY -- 16-31 GB per file)
-----------------------------------------------
    python 04_fetch_soccermon.py --objective objective-TeamB-2020
"""

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

from config import (SUBJECTIVE_DIR, OBJECTIVE_DIR, RAW_DIR, RESULTS_DIR,
                    ZENODO_BASE, ZENODO_FILES, CITATION)

CHUNK = 1 << 20  # 1 MiB


# ─────────────────────────────────────────────────────────────
#  DOWNLOAD
# ─────────────────────────────────────────────────────────────

def _human(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} TB"


def download(key: str, dest_dir: Path, force: bool = False) -> Path:
    if key not in ZENODO_FILES:
        raise KeyError(f"Unknown archive {key!r}. Options: {list(ZENODO_FILES)}")

    fname, approx_size, md5 = ZENODO_FILES[key]
    url = f"{ZENODO_BASE}/{fname}?download=1"
    out = dest_dir / fname

    if out.exists() and not force:
        print(f"[skip] {fname} already present ({_human(out.stat().st_size)}).")
        return out

    free = shutil.disk_usage(dest_dir).free
    if free < approx_size * 2.2:
        print(f"[WARN] {fname} is ~{_human(approx_size)} and unpacks to roughly "
              f"the same again. Free space: {_human(free)}. "
              f"Recommend at least {_human(approx_size * 2.2)}.")
        if input("      Continue anyway? [y/N] ").strip().lower() != "y":
            sys.exit("Aborted.")

    print(f"[get ] {fname}  (~{_human(approx_size)})")
    got = 0
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", approx_size))
        with open(out, "wb") as fh:
            for chunk in r.iter_content(CHUNK):
                fh.write(chunk)
                got += len(chunk)
                pct = 100 * got / total if total else 0
                print(f"\r       {_human(got)} / {_human(total)}  ({pct:5.1f}%)",
                      end="", flush=True)
    print()

    print("[md5 ] verifying …", end=" ", flush=True)
    h = hashlib.md5()
    with open(out, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    if h.hexdigest() == md5:
        print("ok")
    else:
        print(f"MISMATCH\n       expected {md5}\n       got      {h.hexdigest()}")
        sys.exit("Download corrupted -- delete the file and retry.")

    return out


def unpack(zip_path: Path, dest_dir: Path) -> None:
    print(f"[unzip] {zip_path.name} -> {dest_dir}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    print(f"[unzip] done, {sum(1 for _ in dest_dir.rglob('*') if _.is_file())} files")


# ─────────────────────────────────────────────────────────────
#  INSPECT  —  the important part
# ─────────────────────────────────────────────────────────────

def _peek(path: Path) -> dict:
    """Return {shape, columns, sample} for a csv / json / parquet file."""
    info = {"path": str(path.relative_to(RAW_DIR)), "bytes": path.stat().st_size}
    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, nrows=200)
            full = sum(1 for _ in open(path, encoding="utf-8", errors="ignore")) - 1
            info.update(format="csv", rows=full, columns=list(df.columns),
                        dtypes={c: str(t) for c, t in df.dtypes.items()},
                        sample=df.head(3).to_dict("records"))

        elif path.suffix.lower() == ".json":
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, list):
                info.update(format="json-list", rows=len(raw),
                            columns=sorted({k for r in raw[:500]
                                            if isinstance(r, dict) for k in r}),
                            sample=raw[:2])
            elif isinstance(raw, dict):
                # PMSys per-session exports are often {player: [sessions...]}
                keys = list(raw)
                first = raw[keys[0]] if keys else None
                cols = []
                if isinstance(first, list) and first and isinstance(first[0], dict):
                    cols = sorted(first[0])
                info.update(format="json-dict", rows=len(keys),
                            top_level_keys=keys[:10], columns=cols,
                            sample=str(first)[:600])
            else:
                info.update(format="json-other", sample=str(raw)[:400])

        elif path.suffix.lower() in (".parquet", ".pq"):
            df = pd.read_parquet(path)
            info.update(format="parquet", rows=len(df), columns=list(df.columns),
                        dtypes={c: str(t) for c, t in df.dtypes.items()},
                        sample=df.head(3).to_dict("records"))
        else:
            info.update(format="unknown")
    except Exception as exc:                                   # noqa: BLE001
        info.update(format="ERROR", error=f"{type(exc).__name__}: {exc}")
    return info


def inspect(save_report: bool = False) -> None:
    files = sorted(p for p in RAW_DIR.rglob("*")
                   if p.is_file() and p.suffix.lower()
                   in (".csv", ".json", ".parquet", ".pq"))
    if not files:
        sys.exit(f"Nothing to inspect under {RAW_DIR}. Run --subjective first.")

    report = []
    print("=" * 78)
    print(f"SoccerMon archive inspection — {len(files)} data files")
    print("=" * 78)

    for p in files:
        info = _peek(p)
        report.append(info)
        print(f"\n{info['path']}")
        print(f"  format {info.get('format')}   "
              f"rows {info.get('rows', '?')}   size {_human(info['bytes'])}")
        cols = info.get("columns") or []
        if cols:
            print(f"  columns ({len(cols)}):")
            for c in cols:
                print(f"      {c}")
        if info.get("top_level_keys"):
            print(f"  top-level keys (first 10): {info['top_level_keys']}")
        if info.get("error"):
            print(f"  !! {info['error']}")

    print("\n" + "=" * 78)
    print("DAY-3 TASK: copy the column names above into SCHEMA_MAP in config.py.")
    print("Put the real name FIRST in each candidate list.")
    print("=" * 78)

    if save_report:
        out = RESULTS_DIR / "soccermon_inspection.json"
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nSaved -> {out}")


# ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjective", action="store_true",
                    help="download + unpack subjective.zip (705 kB) -- Track A")
    ap.add_argument("--objective", metavar="KEY",
                    help="download one objective archive (16-31 GB) -- Track B, Day 18+")
    ap.add_argument("--inspect", action="store_true",
                    help="print the real structure of everything downloaded")
    ap.add_argument("--save-report", action="store_true",
                    help="write the inspection to results/soccermon_inspection.json")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    if not any((args.subjective, args.objective, args.inspect)):
        ap.print_help()
        return

    if args.subjective:
        z = download("subjective", RAW_DIR, force=args.force)
        unpack(z, SUBJECTIVE_DIR)
        print("\n" + CITATION + "\n")

    if args.objective:
        print("!! Track B. Confirm you have passed the Day-18 gate in the plan.")
        if input("   Proceed? [y/N] ").strip().lower() != "y":
            sys.exit("Aborted -- good call if Track A is not finished.")
        z = download(args.objective, RAW_DIR, force=args.force)
        unpack(z, OBJECTIVE_DIR / args.objective)

    if args.inspect:
        inspect(save_report=args.save_report)


if __name__ == "__main__":
    main()
