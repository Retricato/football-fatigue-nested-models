#!/usr/bin/env python3
"""
10_gps_aggregate.py — TRACK B. SoccerMon raw GPS -> per-session summaries.

Written against the ACTUAL archive, which is far more tractable than the
99 GB download size suggests.

What the files really contain
-----------------------------
    objective-TeamA-2021/2021/2021-05/2021-05-01/
        2021-05-01-TeamA-<uuid>.parquet

  - the filename encodes BOTH the date and the player UUID, so the
    session -> player-day mapping is free
  - 17 columns, but only `time` and `speed` are needed for external load
  - 100 Hz rows, of which the GPS fields update at 10 Hz (each GPS sample is
    repeated 10x for the IMU rows). We de-duplicate to 10 Hz.
  - `speed` is PROVIDED in m/s. There is no need to derive velocity from
    lat/lon, which removes the hardest and most error-prone part of the job.
  - one team-season is ~3,100 files; at ~0.7 s each that is ~40 minutes.

Heart rate is NOT usable
------------------------
`heart_rate` is either 0 or a constant placeholder (82 in every file
checked). It is not a real HR trace. Do not build features on it; keep
"no HRV or heart-rate data" in your Limitations.

Sessions need trimming
----------------------
Units recorded from switch-on, so a file typically spans ~175 minutes:
~20 min of standing around, activity blocks, a mid-session break, then more
idle. Untrimmed "session duration" is meaningless and untrimmed distance
inflates. This script reports BOTH recorded and active duration, and derives
active time from a rolling speed threshold. Justify the threshold in Methods.

Usage
-----
    python 10_gps_aggregate.py --list                     # inventory, no reads
    python 10_gps_aggregate.py --team objective-TeamA-2021 --limit 20
    python 10_gps_aggregate.py --team objective-TeamA-2021 --workers 4
    python 10_gps_aggregate.py --merge                    # -> player-day, join to daily CSV

Resumable: re-running skips sessions already present in the output CSV.
"""

import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from config import ROOT, REAL_DIR, REAL_DAILY_CSV

# ─────────────────────────────────────────────────────────────
#  THRESHOLDS  —  justify every one of these in Methods
# ─────────────────────────────────────────────────────────────

# Velocity bands. These are common in women's elite football but there is no
# consensus, and absolute thresholds borrowed from men's football are a known
# problem in this literature. Cite your source and run a sensitivity check.
HSR_MS            = 4.44    # 16.0 km/h  high-speed running
SPRINT_MS         = 5.29    # 19.0 km/h  sprint
ACC_MS2           =  2.0    # acceleration event
DEC_MS2           = -2.0    # deceleration event
MIN_EVENT_S       =  0.5    # an event must persist this long to count
MAX_PLAUSIBLE_MS  = 12.0    # above this is a GPS artefact

# Activity detection: a 60 s rolling mean above this counts as "active".
ACTIVE_SPEED_MS   = 0.5
ACTIVE_WINDOW_S   = 60

GPS_HZ            = 10.0    # after de-duplication
FILENAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-(Team[AB]-[0-9a-f\-]+)\.parquet$")


# ─────────────────────────────────────────────────────────────

def parse_name(path: Path):
    m = FILENAME_RE.match(path.name)
    return (m.group(1), m.group(2)) if m else (None, None)


def count_events(sig: np.ndarray, thresh: float, dt: float, above: bool) -> int:
    mask = (sig > thresh) if above else (sig < thresh)
    mask = np.nan_to_num(mask, nan=0).astype(bool)
    need = max(int(round(MIN_EVENT_S / dt)), 1)
    n, run = 0, 0
    for m in mask:
        run = run + 1 if m else 0
        if run == need:
            n += 1
    return n


def summarise(path: Path) -> dict | None:
    date, player = parse_name(path)
    if date is None:
        return None
    try:
        d = pd.read_parquet(path, columns=["time", "speed"])
    except Exception as exc:                                  # noqa: BLE001
        return {"file": path.name, "error": f"{type(exc).__name__}: {exc}"}

    if len(d) < 1000:
        return {"file": path.name, "error": "fewer than 1000 rows"}

    # 100 Hz rows -> 10 Hz GPS samples
    g = d.groupby("time", sort=False)["speed"].mean()
    v = g.to_numpy(dtype=float)
    v[v > MAX_PLAUSIBLE_MS] = np.nan

    t = pd.to_timedelta(pd.Series(g.index), errors="coerce")
    span_s = float((t.max() - t.min()).total_seconds())
    dt = span_s / len(v) if len(v) > 1 and span_s > 0 else 1 / GPS_HZ

    # active period: rolling 60 s mean speed above threshold
    w = max(int(round(ACTIVE_WINDOW_S / dt)), 1)
    roll = pd.Series(v).rolling(w, center=True, min_periods=1).mean().to_numpy()
    active = roll > ACTIVE_SPEED_MS
    if active.sum() < w:
        return {"file": path.name, "error": "no active period detected"}

    va = np.where(active, v, np.nan)
    fin = np.isfinite(va)

    dist   = float(np.nansum(va[fin]) * dt)
    hsr    = float(np.nansum(va[fin & (va > HSR_MS)]) * dt)
    sprint = float(np.nansum(va[fin & (va > SPRINT_MS)]) * dt)

    a = np.concatenate([[0.0], np.diff(v) / dt])
    a = pd.Series(a).rolling(max(int(round(0.3 / dt)), 1),
                             center=True, min_periods=1).mean().to_numpy()
    a_act = np.where(active, a, 0.0)

    active_min = active.sum() * dt / 60
    peak = float(np.nanmax(va[fin]))
    dpm = dist / max(active_min, 1e-9)
    sprint_frac = sprint / dist if dist > 0 else 0.0

    rec = {
        "file": path.name,
        "player_id": player,
        "date": date,
        "recorded_minutes": round(span_s / 60, 2),
        "session_duration_minutes": round(active_min, 2),
        "total_distance_meters": round(dist, 1),
        "high_speed_running_distance": round(hsr, 1),
        "sprint_distance": round(sprint, 1),
        "acceleration_count": count_events(a_act, ACC_MS2, dt, True),
        "deceleration_count": count_events(a_act, DEC_MS2, dt, False),
        "peak_speed_ms": round(peak, 2),
        "mean_speed_ms": round(float(np.nanmean(va[fin])), 3),
        "distance_per_min": round(dpm, 1),
        "sprint_fraction": round(sprint_frac, 4),
        "sample_rate_hz": round(1 / dt, 1),
        "n_gps_samples": int(len(v)),
        "error": None,
    }
    rec["qc_flags"] = ";".join(quality_flags(rec))
    rec["qc_pass"] = int(not rec["qc_flags"])
    return rec


# ─────────────────────────────────────────────────────────────
#  QUALITY CONTROL
#
#  Some files are not football. A unit left running in a car, or a GPS
#  artefact, produces sessions with a 41 km/h peak or 40% of distance above
#  the sprint threshold. Those are physiologically impossible and will
#  distort any model trained on them.
#
#  Flag rather than delete: report how many sessions were excluded and why,
#  and run the headline model with and without them.
# ─────────────────────────────────────────────────────────────

QC_MAX_PEAK_MS       = 10.0   # 36 km/h. The women's 100 m record is ~34 km/h.
QC_MAX_DIST_PER_MIN  = 130.0  # football training is typically 60-110 m/min
QC_MAX_SPRINT_FRAC   = 0.15   # >15% of distance at sprint pace is not football
QC_MIN_ACTIVE_MIN    = 15.0   # too short to characterise a session
QC_MAX_DISTANCE_M    = 16000.0


def quality_flags(r: dict) -> list[str]:
    f = []
    if r["peak_speed_ms"] > QC_MAX_PEAK_MS:
        f.append("implausible_peak_speed")
    if r["distance_per_min"] > QC_MAX_DIST_PER_MIN:
        f.append("implausible_intensity")
    if r["sprint_fraction"] > QC_MAX_SPRINT_FRAC:
        f.append("implausible_sprint_fraction")
    if r["session_duration_minutes"] < QC_MIN_ACTIVE_MIN:
        f.append("too_short")
    if r["total_distance_meters"] > QC_MAX_DISTANCE_M:
        f.append("implausible_distance")
    return f


def _work(p: str) -> dict | None:
    return summarise(Path(p))


# ─────────────────────────────────────────────────────────────

def find_files(team_dir: Path) -> list[Path]:
    return sorted(team_dir.rglob("*.parquet"))


def flush(rows: list[dict], out: Path) -> None:
    """
    Append a batch to the output CSV.

    Long runs are checkpointed rather than written once at the end, so a
    crash, a laptop sleeping, or an impatient Ctrl-C costs you one batch
    instead of the whole archive. Combined with the resume-on-restart logic
    this makes the script safe to run in the background overnight.
    """
    if not rows:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if out.exists():
        df.to_csv(out, mode="a", header=False, index=False)
    else:
        df.to_csv(out, index=False)


def inventory(roots: list[Path]) -> None:
    print(f"{'archive':28s} {'files':>7s} {'players':>8s} {'days':>6s}  span")
    print("-" * 72)
    for r in roots:
        files = find_files(r)
        parsed = [parse_name(f) for f in files]
        dates = sorted({d for d, _ in parsed if d})
        players = {p for _, p in parsed if p}
        span = f"{dates[0]} .. {dates[-1]}" if dates else "-"
        print(f"{r.name:28s} {len(files):>7,} {len(players):>8} "
              f"{len(dates):>6}  {span}")
    print("\nAt roughly 0.7 s per file, one team-season takes ~40 min on one "
          "core, ~10 min with --workers 4.")


def run(team_dir: Path, out: Path, limit: int, workers: int,
        stride: int = 1, checkpoint: int = 50) -> None:
    files = find_files(team_dir)
    if not files:
        sys.exit(f"No parquet files under {team_dir}")

    done = set()
    if out.exists():
        prev = pd.read_csv(out).drop_duplicates("file", keep="last")
        prev.to_csv(out, index=False)
        done = set(prev["file"])
        print(f"Resuming: {len(done):,} sessions already summarised in {out.name}")

    todo = [f for f in files if f.name not in done]
    if stride > 1:
        # every Nth file, so the sample spans the whole season rather than
        # just the opening weeks. Use this to validate before a full run.
        todo = todo[::stride]
    if limit:
        todo = todo[:limit]
    if not todo:
        print("Nothing to do.")
        return

    print(f"Processing {len(todo):,} of {len(files):,} files "
          f"with {workers} worker(s)\n")

    rows, failed = [], 0
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_work, str(f)): f for f in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                if r is None:
                    continue
                if r.get("error"):
                    failed += 1
                else:
                    rows.append(r)
                if i % checkpoint == 0:
                    flush(rows, out)
                    rows = []
                if i % 25 == 0 or i == len(todo):
                    print(f"  {i:>5,}/{len(todo):,}  failed {failed}", flush=True)
    else:
        for i, f in enumerate(todo, 1):
            r = summarise(f)
            if r is None:
                continue
            if r.get("error"):
                failed += 1
                print(f"  [fail] {f.name}: {r['error']}")
            else:
                rows.append(r)
                if i % 10 == 0 or i == len(todo):
                    print(f"  {i:>5,}/{len(todo):,}  "
                          f"last: {r['total_distance_meters']:>7,.0f} m in "
                          f"{r['session_duration_minutes']:>5.1f} min active "
                          f"(of {r['recorded_minutes']:.0f} recorded)",
                          flush=True)
            if i % checkpoint == 0:
                flush(rows, out)
                rows = []

    flush(rows, out)
    if not out.exists():
        sys.exit("No sessions summarised.")

    final = pd.read_csv(out)
    print(f"\n{len(final):,} session summaries on disk ({failed} failed "
          f"this run) -> {out}")
    sanity(final)


def sanity(df: pd.DataFrame) -> None:
    ok = df[df.get("qc_pass", 1) == 1] if "qc_pass" in df.columns else df
    d = ok["total_distance_meters"]

    print("\nSANITY CHECK — elite women's football reference values:")
    print("  training  4,000-7,000 m   |   match  9,000-11,000 m   "
          "|  60-110 m/min")
    print(f"  QC-passing sessions: {len(ok):,} of {len(df):,} "
          f"({len(ok) / max(len(df), 1):.1%})")
    print(f"  distance: median {d.median():,.0f} m   "
          f"IQR {d.quantile(.25):,.0f}-{d.quantile(.75):,.0f} m")
    print(f"  active duration: median {ok['session_duration_minutes'].median():.0f} min "
          f"(recorded median {ok['recorded_minutes'].median():.0f} min)")
    print(f"  intensity: median {ok['distance_per_min'].median():.0f} m/min")
    print(f"  peak speed: median {ok['peak_speed_ms'].median() * 3.6:.1f} km/h, "
          f"max {ok['peak_speed_ms'].max() * 3.6:.1f} km/h")

    if "qc_flags" in df.columns:
        bad = df[df["qc_pass"] == 0]
        if len(bad):
            reasons = (bad["qc_flags"].str.split(";").explode()
                       .value_counts())
            print(f"\n  {len(bad):,} session(s) flagged:")
            for r, n in reasons.items():
                print(f"      {r:32s} {n:>5,}")
            print("  Report this count and the exclusion rule in Methods, and "
                  "rerun the headline model with and without them.")

    if not (3000 < d.median() < 12000):
        print("\n  [WARN] Median distance is outside the plausible range. "
              "Check the trimming threshold before modelling on these.")
    else:
        print("\n  QC-passing values are in the expected range.")


def merge(sessions_csv: Path, daily_csv: Path, out: Path,
          drop_flagged: bool = True) -> None:
    s = pd.read_csv(sessions_csv)
    if "error" in s.columns:
        s = s[s["error"].isna()]
    if drop_flagged and "qc_pass" in s.columns:
        n0 = len(s)
        s = s[s["qc_pass"] == 1]
        print(f"Excluded {n0 - len(s):,} QC-flagged sessions "
              f"({len(s):,} remain). Use --keep-flagged to include them.")
    s["date"] = pd.to_datetime(s["date"])

    # several units can be worn in one day -> aggregate to player-day
    agg = s.groupby(["player_id", "date"], as_index=False).agg(
        total_distance_meters=("total_distance_meters", "sum"),
        high_speed_running_distance=("high_speed_running_distance", "sum"),
        sprint_distance=("sprint_distance", "sum"),
        acceleration_count=("acceleration_count", "sum"),
        deceleration_count=("deceleration_count", "sum"),
        gps_active_minutes=("session_duration_minutes", "sum"),
        peak_speed_ms=("peak_speed_ms", "max"),
        n_gps_sessions=("file", "count"))

    daily = pd.read_csv(daily_csv, parse_dates=["date"])
    merged = daily.merge(agg, on=["player_id", "date"], how="left")

    have = merged["total_distance_meters"].notna()
    print(f"GPS coverage: {have.sum():,} of {len(merged):,} player-days "
          f"({have.mean():.1%})")
    both = merged[have & (merged["n_sessions"] > 0)]
    print(f"  player-days with BOTH a GPS file and a logged session: {len(both):,}")
    if len(both) > 30:
        r = both["total_distance_meters"].corr(both["session_load_au"])
        print(f"  correlation, GPS distance vs. sRPE: r = {r:.3f}")
        print("  External and internal load should correlate moderately "
              "(r ~ 0.4-0.7). A very low value means the trimming or the "
              "player-day join is wrong.")

    merged.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print("\nNext: rebuild labels and rerun training with GPS features:")
    print(f"  python 07_labels.py --data {out} "
          f"--out data/real/soccermon_model_ready_gps.csv")
    print("  python 08_train_real.py --data data/real/soccermon_model_ready_gps.csv "
          "--include-gps --models rf xgb --cv grouped lopo")


# ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", default="objective-TeamA-2021",
                    help="archive folder name, or a path to it")
    ap.add_argument("--list", action="store_true",
                    help="inventory every objective-* folder and exit")
    ap.add_argument("--limit", type=int, default=0, help="process only N files")
    ap.add_argument("--stride", type=int, default=1,
                    help="take every Nth file, spread across the season "
                         "(validate before committing to a full run)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--checkpoint", type=int, default=50,
                    help="append results to the CSV every N files")
    ap.add_argument("--out", default=str(REAL_DIR / "gps_sessions.csv"))
    ap.add_argument("--merge", action="store_true",
                    help="aggregate to player-day and join onto the daily CSV")
    # Merge onto the COHORT file, not the raw daily file. Merging onto the
    # raw file would silently readmit the low-compliance players the cohort
    # rule excluded, putting Track A and Track B on different player bases
    # and confounding any comparison between them.
    ap.add_argument("--daily", default=str(REAL_DIR / "soccermon_cohort.csv"))
    ap.add_argument("--merged-out", default=str(REAL_DIR / "soccermon_daily_gps.csv"))
    ap.add_argument("--keep-flagged", action="store_true",
                    help="include QC-flagged sessions in the merge")
    args = ap.parse_args()

    if args.list:
        roots = sorted(p for p in ROOT.iterdir()
                       if p.is_dir() and p.name.startswith("objective-"))
        if not roots:
            sys.exit("No objective-* folders found in the project root.")
        inventory(roots)
        return

    if args.merge:
        merge(Path(args.out), Path(args.daily), Path(args.merged_out),
              drop_flagged=not args.keep_flagged)
        return

    team_dir = Path(args.team)
    if not team_dir.is_absolute() and not team_dir.exists():
        team_dir = ROOT / args.team
    if not team_dir.exists():
        sys.exit(f"{team_dir} not found. Run --list to see what is available.")

    run(team_dir, Path(args.out), args.limit, args.workers, args.stride,
        args.checkpoint)


if __name__ == "__main__":
    main()
