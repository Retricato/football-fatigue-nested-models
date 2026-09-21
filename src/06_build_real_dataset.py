#!/usr/bin/env python3
"""
06_build_real_dataset.py — SoccerMon subjective archive -> tidy daily CSV.

Written against the ACTUAL archive layout, not a guess at it:

  wellness/*.csv        WIDE  rows = dates (dd.mm.yyyy), cols = 50 player UUIDs
  training-load/*.csv   WIDE  same shape, pre-computed daily metrics
  training-load/session.json
                        LONG  {player: [{srpe, rpe, duration, date}, ...]}
  injury/injury.csv     LONG  player_name, type = JSON {body_part: severity}
  illness/illness.csv   LONG  player_name, problems = JSON list
  game-performance/…    LONG  player_name, 3 ratings, timestamp

Pipeline
--------
  1. melt every wide csv to long (date, player_id, value)
  2. aggregate session.json to daily: sum sRPE and duration, max RPE, count
  3. join the event logs
  4. build the calendar spine so rest days exist as rows
  5. distinguish rest days (load = 0) from missing wellness (NaN)
  6. derived features -- ACWR, rolling load, EWMA fatigue, monotony, strain,
     Hooper index -- all with TRAILING windows only

Usage
-----
    python 06_build_real_dataset.py                 # steps 1-5
    python 06_build_real_dataset.py --features      # steps 1-6
    python 06_build_real_dataset.py --check-polarity  # verify scale directions
    python 06_build_real_dataset.py --dry-run       # inventory the archive
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import (SUBJECTIVE_DIR, REAL_DAILY_CSV, DATE_FORMAT,
                    WELLNESS_FILES, TRAINING_LOAD_FILES,
                    RESCALE_1_5_TO_1_10, HOOPER_COMPONENTS)


# ─────────────────────────────────────────────────────────────
#  WIDE -> LONG
# ─────────────────────────────────────────────────────────────

def melt_wide(path: Path, value_name: str) -> pd.DataFrame:
    """
    Wide SoccerMon csv -> long (date, player_id, <value_name>).

    The first column is the date but its header varies by file
    ('Date', 'Fatigue Data', 'SleepDurH Data', ...), so it is taken
    positionally rather than by name.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], format=DATE_FORMAT, errors="coerce")

    player_cols = [c for c in df.columns if c != "date"]
    long = df.melt("date", value_vars=player_cols,
                   var_name="player_id", value_name=value_name)
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
    return long.dropna(subset=["date"])


def load_wide_group(folder: Path, file_map: dict[str, str],
                    label: str) -> pd.DataFrame:
    out, missing = None, []
    for fname, col in file_map.items():
        p = folder / fname
        if not p.exists():
            missing.append(fname)
            continue
        m = melt_wide(p, col)
        out = m if out is None else out.merge(m, on=["date", "player_id"],
                                              how="outer")
        print(f"  [{label:9s}] {fname:22s} -> {col:22s} "
              f"{m[col].notna().sum():>7,} values")
    if missing:
        print(f"  [{label:9s}] MISSING FILES: {missing}")
    return out if out is not None else pd.DataFrame(
        columns=["date", "player_id"])


# ─────────────────────────────────────────────────────────────
#  SESSIONS  (the only long-format training-load source)
# ─────────────────────────────────────────────────────────────

def load_sessions(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = [{"player_id": p, "date": r.get("date"),
             "srpe": r.get("srpe"), "rpe": r.get("rpe"),
             "duration": r.get("duration")}
            for p, sessions in raw.items() for r in sessions]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format=DATE_FORMAT, errors="coerce")
    for c in ("srpe", "rpe", "duration"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"  [sessions ] session.json           "
          f"{len(df):>7,} sessions, {df['player_id'].nunique()} players")
    return df.dropna(subset=["date"])


def aggregate_sessions_daily(sess: pd.DataFrame) -> pd.DataFrame:
    """
    Multiple sessions per day -> one row.

    Decisions, to be stated in Methods:
      sRPE and duration are SUMMED (total daily load)
      RPE is the MAX across the day (peak perceived intensity)
      n_sessions is retained as a feature in its own right

    Up to 6 sessions occur on a single day in this archive.
    """
    daily = sess.groupby(["player_id", "date"], as_index=False).agg(
        session_load_au=("srpe", "sum"),
        session_duration_minutes=("duration", "sum"),
        RPE=("rpe", "max"),
        n_sessions=("srpe", "size"),
    )
    return daily


# ─────────────────────────────────────────────────────────────
#  EVENT LOGS
# ─────────────────────────────────────────────────────────────

def load_injuries(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        try:
            payload = json.loads(r["type"])
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        sev = list(payload.values())
        rows.append({
            "player_id": r["player_name"],
            "date": r["timestamp"],
            "injury_flag": 1,
            "n_injuries": len(payload),
            "injury_major": int(any(str(s).lower() == "major" for s in sev)),
            "injury_sites": ";".join(payload.keys()),
        })
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"], format=DATE_FORMAT, errors="coerce")
    agg = out.groupby(["player_id", "date"], as_index=False).agg(
        injury_flag=("injury_flag", "max"),
        n_injuries=("n_injuries", "sum"),
        injury_major=("injury_major", "max"))
    print(f"  [injury   ] injury.csv             "
          f"{len(df):>7,} reports, {int(agg['n_injuries'].sum())} injuries, "
          f"{agg['player_id'].nunique()} players")
    return agg


def load_illness(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        try:
            problems = json.loads(r["problems"])
        except (TypeError, ValueError, json.JSONDecodeError):
            problems = []
        rows.append({"player_id": r["player_name"], "date": r["timestamp"],
                     "illness_flag": 1, "n_symptoms": len(problems)})
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"], format=DATE_FORMAT, errors="coerce")
    agg = out.groupby(["player_id", "date"], as_index=False).agg(
        illness_flag=("illness_flag", "max"), n_symptoms=("n_symptoms", "sum"))
    print(f"  [illness  ] illness.csv            {len(df):>7,} reports")
    return agg


def load_game_performance(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={"player_name": "player_id", "timestamp": "date"})
    df["date"] = pd.to_datetime(df["date"], format=DATE_FORMAT, errors="coerce")
    df["is_match_day"] = 1
    keep = ["player_id", "date", "is_match_day", "team_performance",
            "offensive_performance", "defensive_performance"]
    out = df[[c for c in keep if c in df.columns]]
    agg = out.groupby(["player_id", "date"], as_index=False).max()
    print(f"  [game     ] game-performance.csv   {len(df):>7,} reports")
    return agg


# ─────────────────────────────────────────────────────────────
#  RESCALING AND POLARITY
# ─────────────────────────────────────────────────────────────

def rescale_wellness(df: pd.DataFrame) -> pd.DataFrame:
    """
    PMSys items are 1-5 with HIGHER = BETTER for every item.

    We rescale to 1-10 for comparability with the synthetic schema, and flip
    fatigue / soreness / stress so the resulting column names mean what they
    say: high `muscle_soreness` = sore. See the polarity note in config.py.
    """
    df = df.copy()
    for raw, (new, flip) in RESCALE_1_5_TO_1_10.items():
        if raw not in df.columns:
            continue
        v = pd.to_numeric(df[raw], errors="coerce")
        if flip:
            v = 6 - v                      # 1-5 reversed
        df[new] = (v - 1) * 9 / 4 + 1      # 1-5 -> 1-10
    return df


def check_polarity(well: pd.DataFrame, load_daily: pd.DataFrame) -> None:
    """Verify the scale directions empirically instead of trusting a docstring."""
    d = well.merge(load_daily, on=["player_id", "date"], how="left")
    raw_cols = [c for c in ("fatigue_raw", "mood_raw", "sleep_quality_raw",
                            "soreness_raw", "stress_raw", "sleep_hours")
                if c in d.columns]

    print("\nPOLARITY CHECK")
    print("  Correlation of each raw 1-5 item with readiness (1-10, "
          "unambiguously higher = more ready):")
    for c in raw_cols:
        r = d[c].corr(d["readiness"])
        direction = "higher = BETTER" if r > 0 else "higher = WORSE"
        print(f"    {c:20s} r = {r:+.3f}   -> {direction}")

    d = d.sort_values(["player_id", "date"])
    d["load_prev"] = d.groupby("player_id")["session_load_au"].shift(1)
    print("\n  Correlation with PREVIOUS day's training load "
          "(a hard day should worsen next-day state):")
    for c in raw_cols + (["readiness"] if "readiness" in d.columns else []):
        r = d[c].corr(d["load_prev"])
        print(f"    {c:20s} r = {r:+.3f}")

    print("\n  Expected: all items correlate POSITIVELY with readiness and "
          "NEGATIVELY with previous-day load. If that is what you see, the "
          "flips in config.RESCALE_1_5_TO_1_10 are correct. If it is not, "
          "stop and fix them -- getting this backwards silently inverts "
          "every result in the thesis.")


# ─────────────────────────────────────────────────────────────
#  SPINE AND REST/MISSING
# ─────────────────────────────────────────────────────────────

def build_spine(players, start, end) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="D")
    return pd.DataFrame([(p, d) for p in players for d in idx],
                        columns=["player_id", "date"])


def mark_rest_vs_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rest day : no session logged        -> load = 0, n_sessions = 0
    Missing  : no wellness report       -> stays NaN, never zero-filled

    Conflating the two is the most common error in this literature: it turns
    non-compliance into fabricated 'perfect wellness' days.
    """
    for c in ("session_load_au", "session_duration_minutes", "n_sessions"):
        if c in df.columns:
            df[c] = df[c].fillna(0)
    if "RPE" in df.columns:
        df["RPE"] = df["RPE"].fillna(0)

    df["is_rest_day"] = (df["n_sessions"] == 0).astype(int)

    wcols = [new for _, (new, _) in RESCALE_1_5_TO_1_10.items()
             if new in df.columns]
    df["wellness_reported"] = (df[wcols].notna().any(axis=1).astype(int)
                               if wcols else 0)

    for c in ("injury_flag", "illness_flag", "is_match_day",
              "n_injuries", "injury_major", "n_symptoms"):
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)
    return df


# ─────────────────────────────────────────────────────────────
#  DERIVED FEATURES  (trailing windows only -- no look-ahead)
# ─────────────────────────────────────────────────────────────

def derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors compute_derived_features() in 01_data_generator.py so the two
    domains stay comparable, but driven by sRPE rather than a distance proxy."""
    out = []
    for pid, g in df.groupby("player_id", sort=False):
        g = g.sort_values("date").copy()
        load = g["session_load_au"].fillna(0)

        g["7_day_workload_average"]  = load.rolling(7,  min_periods=1).mean()
        g["28_day_workload_average"] = load.rolling(28, min_periods=1).mean()

        acute   = load.rolling(7,  min_periods=3).mean()
        chronic = load.rolling(28, min_periods=7).mean()
        g["ACWR"] = (acute / chronic.replace(0, np.nan)).round(3)

        # EWMA span 14 -- identical to the synthetic generator
        g["cumulative_fatigue_score"] = load.ewm(span=14, adjust=False).mean().round(3)

        # Foster monotony and strain over a trailing week
        wk_mean = load.rolling(7, min_periods=3).mean()
        wk_std  = load.rolling(7, min_periods=3).std()
        g["monotony"] = (wk_mean / wk_std.replace(0, np.nan)).round(3)
        g["strain"]   = (load.rolling(7, min_periods=3).sum() * g["monotony"]).round(1)

        # Hooper Index -- higher = worse. Weights encode direction; see config.
        comp = [w * g[c] for c, w in HOOPER_COMPONENTS.items() if c in g.columns]
        if comp:
            n_neg = sum(1 for c, w in HOOPER_COMPONENTS.items()
                        if w < 0 and c in g.columns)
            g["hooper_index"] = (sum(comp) + n_neg * 11).round(2)
            g["7_day_hooper_trend"] = (
                g["hooper_index"].rolling(7, min_periods=3)
                .apply(lambda s: np.polyfit(np.arange(len(s)), s, 1)[0]
                       if s.notna().sum() >= 3 else np.nan, raw=False).round(3))
        else:
            g["hooper_index"] = np.nan
            g["7_day_hooper_trend"] = np.nan

        dsr, count = [], 0
        for rest in g["is_rest_day"].to_numpy():
            count = 0 if rest else count + 1
            dsr.append(count)
        g["days_since_rest"] = dsr

        g["day_number"]  = np.arange(1, len(g) + 1)
        g["week_number"] = ((g["day_number"] - 1) // 7) + 1
        out.append(g)

    res = pd.concat(out, ignore_index=True)

    # cross-checks against the authors' own columns
    if "acwr_provided" in res.columns:
        both = res[["ACWR", "acwr_provided"]].dropna()
        both = both[both["acwr_provided"] > 0]
        if len(both) > 30:
            print(f"\n  ACWR cross-check vs. the authors' column: "
                  f"r = {both['ACWR'].corr(both['acwr_provided']):.3f} "
                  f"(n = {len(both):,})")
            print("    Perfect agreement is not expected: theirs is capped at "
                  "4.0 and uses a different smoothing. Document which you use.")

    if "daily_load_provided" in res.columns:
        both = res[["session_load_au", "daily_load_provided"]].dropna()
        exact = (both["session_load_au"] - both["daily_load_provided"]).abs().lt(1e-6).mean()
        print(f"  Daily-load cross-check: our session.json aggregation matches "
              f"the authors' daily_load exactly on {exact:.1%} of player-days")
        if exact < 0.95:
            print("    [WARN] Below 95% -- investigate before modelling.")
    return res


# ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(SUBJECTIVE_DIR))
    ap.add_argument("--out", default=str(REAL_DAILY_CSV))
    ap.add_argument("--features", action="store_true",
                    help="also compute derived features")
    ap.add_argument("--check-polarity", action="store_true",
                    help="verify wellness scale directions and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="inventory the archive and exit")
    args = ap.parse_args()

    src = Path(args.src)
    if not (src / "wellness").is_dir():
        sys.exit(f"{src} does not look like the SoccerMon subjective archive.\n"
                 f"Expected subfolders: wellness/ training-load/ injury/ "
                 f"illness/ game-performance/")

    if args.dry_run:
        print(f"Archive at {src}\n")
        for sub in sorted(p for p in src.iterdir() if p.is_dir()):
            print(f"  {sub.name}/")
            for f in sorted(sub.iterdir()):
                print(f"      {f.name:26s} {f.stat().st_size / 1024:>9,.1f} kB")
        return

    print(f"READING {src}\n")
    well = load_wide_group(src / "wellness", WELLNESS_FILES, "wellness")
    tl   = load_wide_group(src / "training-load", TRAINING_LOAD_FILES, "load")
    sess = load_sessions(src / "training-load" / "session.json")
    load_daily = aggregate_sessions_daily(sess)
    inj  = load_injuries(src / "injury" / "injury.csv")
    ill  = load_illness(src / "illness" / "illness.csv")
    game = load_game_performance(src / "game-performance" / "game-performance.csv")

    well = rescale_wellness(well)

    if args.check_polarity:
        check_polarity(well, load_daily)
        return

    players = sorted(set(well["player_id"]) | set(load_daily["player_id"]))
    all_dates = pd.concat([well["date"], load_daily["date"]])
    spine = build_spine(players, all_dates.min(), all_dates.max())
    print(f"\nSPINE  {len(players)} players x "
          f"{all_dates.min().date()} to {all_dates.max().date()} "
          f"= {len(spine):,} player-days")

    df = spine
    for part, name in ((load_daily, "sessions"), (well, "wellness"),
                       (tl, "training-load"), (inj, "injury"),
                       (ill, "illness"), (game, "game")):
        if part is not None and not part.empty:
            df = df.merge(part, on=["player_id", "date"], how="left")

    df = mark_rest_vs_missing(df.sort_values(["player_id", "date"])
                                .reset_index(drop=True))

    # team label comes free from the anonymised UUID prefix
    df["team"] = df["player_id"].str.split("-").str[0]

    # short, stable, publication-friendly codes (A01, B07, ...). The full UUID
    # stays in player_id; use player_code in figures and tables.
    codes = {}
    for team, grp in df.groupby("team"):
        for i, pid in enumerate(sorted(grp["player_id"].unique()), 1):
            codes[pid] = f"{team.replace('Team', '')}{i:02d}"
    df["player_code"] = df["player_id"].map(codes)
    pd.DataFrame(sorted(codes.items()), columns=["player_id", "player_code"]) \
        .to_csv(Path(args.out).parent / "player_code_lookup.csv", index=False)

    if args.features:
        print("\nDERIVED FEATURES")
        df = derived_features(df)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nWROTE {args.out}")
    print(f"  {len(df):,} rows x {df.shape[1]} cols")
    print(f"  {df['player_id'].nunique()} players "
          f"({df.groupby('team')['player_id'].nunique().to_dict()})")
    print(f"  {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  wellness reported : {int(df['wellness_reported'].sum()):>7,} "
          f"({df['wellness_reported'].mean():.1%} of player-days)")
    print(f"  training sessions : {int(df['n_sessions'].sum()):>7,}")
    print(f"  rest days         : {int(df['is_rest_day'].sum()):>7,}")
    for c, lab in (("injury_flag", "injury days"),
                   ("illness_flag", "illness days"),
                   ("is_match_day", "match days")):
        if c in df.columns:
            print(f"  {lab:18s}: {int(df[c].sum()):>7,}")
    print("\nNext: python 05_real_data_audit.py --apply-cohort")


if __name__ == "__main__":
    main()
