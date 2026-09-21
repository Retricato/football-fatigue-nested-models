#!/usr/bin/env python3
"""
12_recovery_dynamics.py — how long does it take a player to recover?

Your title promises "Fatigue and Recovery Dynamics". This is the recovery half.
It also answers the part of proposal RQ3 that asks about individualized recovery
estimation.

Method
------
1. Identify high-load days: session load above the player's own 80th percentile.
2. From each such day, follow the Hooper index forward.
3. Recovery time = days until the Hooper index returns to at or below the
   player's own trailing baseline.
4. Compare recovery time between players, and against load magnitude.

Everything is per-player, for the same reason the labels are: Hooper scores are
not comparable between athletes.

Usage
-----
    python 12_recovery_dynamics.py
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from config import REAL_MODEL_CSV, RESULTS_DIR, FIG_DIR

MAX_FOLLOW = 10          # give up after this many days
LOAD_PCTILE = 80         # a "high load" day for that player


def recovery_events(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, g in df.groupby("player_id", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        if g["hooper_index"].notna().sum() < 60:
            continue

        # the player's own reference points
        load_cut = g["session_load_au"].quantile(LOAD_PCTILE / 100)
        baseline = g["hooper_index"].rolling(28, min_periods=14).mean().shift(1)

        code = g["player_code"].iloc[0] if "player_code" in g else pid
        hi = g.index[(g["session_load_au"] >= load_cut) & (load_cut > 0)]

        for i in hi:
            base = baseline.iloc[i]
            if not np.isfinite(base):
                continue
            # only count events where she was actually elevated the next day
            nxt = g["hooper_index"].iloc[i + 1] if i + 1 < len(g) else np.nan
            if not np.isfinite(nxt) or nxt <= base:
                continue

            days, peak = np.nan, nxt
            for k in range(1, MAX_FOLLOW + 1):
                j = i + k
                if j >= len(g):
                    break
                v = g["hooper_index"].iloc[j]
                if not np.isfinite(v):
                    continue
                peak = max(peak, v)
                if v <= base:
                    days = k
                    break

            rows.append({
                "player_id": pid, "player_code": code,
                "date": g["date"].iloc[i],
                "load": g["session_load_au"].iloc[i],
                "load_z": (g["session_load_au"].iloc[i] - g["session_load_au"].mean())
                          / (g["session_load_au"].std() + 1e-9),
                "baseline": base,
                "peak_elevation": peak - base,
                "recovery_days": days,
                "censored": int(not np.isfinite(days)),
                "acwr": g["ACWR"].iloc[i] if "ACWR" in g else np.nan,
                "gps_distance": g["total_distance_meters"].iloc[i]
                                if "total_distance_meters" in g else np.nan,
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(REAL_MODEL_CSV))
    args = ap.parse_args()

    df = pd.read_csv(args.data, parse_dates=["date"])
    ev = recovery_events(df)
    if ev.empty:
        raise SystemExit("No recovery events found.")

    done = ev[ev.censored == 0]
    print(f"High-load events analysed : {len(ev):,} across {ev.player_code.nunique()} players")
    print(f"Recovered within {MAX_FOLLOW} days   : {len(done):,} ({1 - ev.censored.mean():.1%})")
    print(f"\nRECOVERY TIME (days to return to own baseline)")
    print(f"  median {done.recovery_days.median():.1f}   "
          f"mean {done.recovery_days.mean():.2f}   "
          f"IQR {done.recovery_days.quantile(.25):.0f}-{done.recovery_days.quantile(.75):.0f}")
    dist = done.recovery_days.value_counts().sort_index()
    print("\n  days  events   share")
    for d, n in dist.items():
        print(f"  {int(d):>4}  {n:>6,}  {n/len(done):>6.1%}")

    # between-player variation
    per = (done.groupby("player_code")["recovery_days"]
               .agg(["median", "mean", "count"])
               .query("count >= 10").sort_values("median"))
    print(f"\nBETWEEN-PLAYER VARIATION  ({len(per)} players with >=10 events)")
    print(f"  median recovery ranges {per['median'].min():.1f} to {per['median'].max():.1f} days")
    print(f"  mean   recovery ranges {per['mean'].min():.2f} to {per['mean'].max():.2f} days")

    # does a harder session take longer to recover from?
    sub = done.dropna(subset=["load_z"])
    r, p = stats.spearmanr(sub.load_z, sub.recovery_days)
    print(f"\nLOAD MAGNITUDE vs RECOVERY TIME")
    print(f"  Spearman rho = {r:+.3f}  (p = {p:.3g}, n = {len(sub):,})")

    r2, p2 = stats.spearmanr(sub.peak_elevation, sub.recovery_days)
    print(f"  peak elevation vs recovery time: rho = {r2:+.3f} (p = {p2:.3g})")

    out = {
        "n_events": int(len(ev)),
        "n_recovered": int(len(done)),
        "recovered_share": round(float(1 - ev.censored.mean()), 4),
        "median_recovery_days": float(done.recovery_days.median()),
        "mean_recovery_days": round(float(done.recovery_days.mean()), 3),
        "player_median_min": float(per["median"].min()),
        "player_median_max": float(per["median"].max()),
        "spearman_load_vs_recovery": {"rho": round(float(r), 4), "p": float(p)},
        "spearman_elevation_vs_recovery": {"rho": round(float(r2), 4), "p": float(p2)},
        "distribution": {int(k): int(v) for k, v in dist.items()},
    }
    (RESULTS_DIR / "recovery_dynamics.json").write_text(json.dumps(out, indent=2))
    ev.to_csv(RESULTS_DIR / "recovery_events.csv", index=False)

    # ---- figure -------------------------------------------------------
    outdir = FIG_DIR / "real"; outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax[0].bar(dist.index, dist.values / len(done), color="#2980b9", alpha=.85)
    ax[0].axvline(done.recovery_days.median(), ls="--", c="#c0392b",
                  label=f"median {done.recovery_days.median():.0f} days")
    ax[0].set_xlabel("days to return to baseline")
    ax[0].set_ylabel("share of high-load events")
    ax[0].set_title("Recovery time after a high-load session")
    ax[0].legend(fontsize=9)

    ax[1].barh(range(len(per)), per["median"], color="#27ae60", alpha=.85)
    ax[1].set_yticks(range(len(per)))
    ax[1].set_yticklabels(per.index, fontsize=6)
    ax[1].set_xlabel("median recovery time (days)")
    ax[1].set_title("Recovery time varies between athletes")
    fig.tight_layout()
    fig.savefig(outdir / "figR11_recovery.png", dpi=200)
    plt.close(fig)

    print(f"\nfigure -> {outdir/'figR11_recovery.png'}")
    print(f"results -> {RESULTS_DIR/'recovery_dynamics.json'}")


if __name__ == "__main__":
    main()
