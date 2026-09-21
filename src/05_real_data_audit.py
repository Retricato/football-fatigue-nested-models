#!/usr/bin/env python3
"""
05_real_data_audit.py — coverage, compliance and missingness audit.

Run this on the daily dataset produced by 06_build_real_dataset.py.
Its outputs are not just housekeeping: real-world adherence to daily
wellness reporting is a genuine finding and belongs in your Results
chapter, not buried in an appendix.

Usage
-----
    python 05_real_data_audit.py
    python 05_real_data_audit.py --data data/real/soccermon_daily.csv
    python 05_real_data_audit.py --apply-cohort      # writes the filtered cohort
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import (REAL_DAILY_CSV, REAL_DIR, FIG_DIR, RESULTS_DIR,
                    COHORT_RULES)


def per_player_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per player: span, coverage, compliance, streaks."""
    rows = []
    wellness_cols = [c for c in ("fatigue_score", "mood_score", "sleep_quality",
                                 "muscle_soreness", "stress_score", "readiness")
                     if c in df.columns]

    for pid, g in df.groupby("player_id"):
        g = g.sort_values("date")
        span = (g["date"].max() - g["date"].min()).days + 1

        wellness_days = int(g[wellness_cols].notna().any(axis=1).sum()) if wellness_cols else 0
        load_days = int((g.get("n_sessions", pd.Series(0, index=g.index)) > 0).sum())

        # longest run of consecutive days with a wellness report
        has = g[wellness_cols].notna().any(axis=1).to_numpy() if wellness_cols else np.array([])
        streak = best = 0
        for v in has:
            streak = streak + 1 if v else 0
            best = max(best, streak)

        rows.append({
            "player_id": pid,
            "first_date": g["date"].min().date(),
            "last_date": g["date"].max().date(),
            "span_days": span,
            "rows": len(g),
            "wellness_days": wellness_days,
            "wellness_compliance": round(wellness_days / span, 3) if span else 0,
            "load_sessions": int(g.get("n_sessions", pd.Series(0, index=g.index)).sum()),
            "load_days": load_days,
            "longest_wellness_streak": best,
        })

    return pd.DataFrame(rows).sort_values("wellness_days", ascending=False)


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().mean().sort_values(ascending=False)
    return (miss.rename("missing_fraction").to_frame()
                .assign(present=(1 - miss).round(3))
                .round(3))


def plot_missingness_heatmap(df: pd.DataFrame, out: str) -> None:
    num = df.select_dtypes(include="number")
    if num.empty:
        return
    pivot = (df.assign(_m=num.isna().mean(axis=1))
               .pivot_table(index="player_id", columns=df["date"].dt.to_period("M")
                            .astype(str), values="_m", aggfunc="mean"))
    fig, ax = plt.subplots(figsize=(max(10, pivot.shape[1] * 0.5),
                                    max(5, pivot.shape[0] * 0.28)))
    sns.heatmap(pivot, cmap="rocket_r", vmin=0, vmax=1, ax=ax,
                cbar_kws={"label": "fraction of fields missing"})
    ax.set_title("Data completeness by player and month\n"
                 "(darker = more missing; gaps are real-world reporting behaviour)")
    ax.set_xlabel("month")
    ax.set_ylabel("player")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  figure -> {out}")


def plot_compliance_over_time(df: pd.DataFrame, out: str) -> None:
    wellness_cols = [c for c in ("fatigue_score", "mood_score", "sleep_quality",
                                 "muscle_soreness", "stress_score")
                     if c in df.columns]
    if not wellness_cols:
        return
    d = (df.assign(reported=df[wellness_cols].notna().any(axis=1))
           .groupby(df["date"].dt.to_period("W").dt.start_time)["reported"]
           .mean().rename("compliance"))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(d.index, d.values, lw=1.6, color="#c0392b")
    ax.axhline(d.mean(), ls="--", lw=1, color="grey",
               label=f"season mean = {d.mean():.2f}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of squad reporting wellness")
    ax.set_xlabel("week")
    ax.set_title("Weekly wellness-reporting compliance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  figure -> {out}")


def select_cohort(summary: pd.DataFrame, rules: dict) -> list:
    keep = summary[
        (summary["wellness_days"] >= rules["min_wellness_days"]) &
        (summary["load_sessions"] >= rules["min_load_sessions"])
    ]["player_id"].tolist()
    return keep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(REAL_DAILY_CSV))
    ap.add_argument("--apply-cohort", action="store_true",
                    help="write data/real/soccermon_cohort.csv with the filter applied")
    args = ap.parse_args()

    df = pd.read_csv(args.data, parse_dates=["date"])
    print(f"Loaded {len(df):,} rows, {df['player_id'].nunique()} players, "
          f"{df['date'].min().date()} to {df['date'].max().date()}\n")

    summary = per_player_summary(df)
    print("PER-PLAYER COVERAGE")
    print(summary.to_string(index=False))
    summary.to_csv(RESULTS_DIR / "audit_per_player.csv", index=False)

    print("\nMISSINGNESS BY VARIABLE")
    miss = missingness_table(df)
    print(miss.to_string())
    miss.to_csv(RESULTS_DIR / "audit_missingness.csv")

    FIG_DIR.joinpath("real").mkdir(parents=True, exist_ok=True)
    plot_missingness_heatmap(df, str(FIG_DIR / "real" / "figR1_missingness.png"))
    plot_compliance_over_time(df, str(FIG_DIR / "real" / "figR2_compliance.png"))

    cohort = select_cohort(summary, COHORT_RULES)
    print(f"\nCOHORT RULE  {COHORT_RULES}")
    print(f"  players meeting the rule: {len(cohort)} of {len(summary)}")
    if len(cohort) < 20:
        print("  [WARN] Fewer than 20 players survive. Relax min_wellness_days in "
              "config.py to 100 and re-run, then report the relaxation in Methods.")
    print(f"  {cohort}")

    if args.apply_cohort:
        out = REAL_DIR / "soccermon_cohort.csv"
        df[df["player_id"].isin(cohort)].to_csv(out, index=False)
        print(f"\nWrote cohort -> {out}")

    print("\nReminder: report compliance and cohort attrition in Results. "
          "Missingness is a finding, not only a nuisance.")


if __name__ == "__main__":
    main()
