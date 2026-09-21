#!/usr/bin/env python3
"""
02b_eda_real.py — EDA for the real (SoccerMon) dataset.

`02_eda_visualization.py` is tightly coupled to GPS and HRV columns that do
not exist in SoccerMon, so it skips almost every figure on real data. This is
its counterpart: the same analytical intent, built from the variables that
actually exist.

Figures produced
----------------
  figR4  correlation matrix over available features
  figR5  squad training load across the season, with match/rest structure
  figR6  Hooper index vs. training load for a sample of players
  figR7  wellness component trends, squad mean with between-player spread
  figR8  ACWR distribution and its relationship to next-day risk
  figR9  between-player variation in wellness reporting (motivates per-player
         thresholds -- this is the figure that justifies your label design)

Usage
-----
    python 02b_eda_real.py
    python 02b_eda_real.py --data data/real/soccermon_cohort.csv --out figures/real
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import REAL_MODEL_CSV, FIG_DIR

sns.set_style("whitegrid")
PAL = {"load": "#2c3e50", "risk": "#c0392b", "ok": "#27ae60", "acc": "#2980b9"}

WELLNESS = ["fatigue_score", "mood_score", "sleep_quality",
            "muscle_soreness", "stress_score", "readiness", "sleep_hours"]
LOAD = ["RPE", "session_duration_minutes", "session_load_au", "n_sessions",
        "ACWR", "7_day_workload_average", "28_day_workload_average",
        "cumulative_fatigue_score", "monotony", "strain"]


def _avail(df, cols):
    return [c for c in cols if c in df.columns and df[c].notna().any()]


def fig_correlation(df, out):
    cols = _avail(df, LOAD + WELLNESS + ["hooper_index", "days_since_rest"])
    if len(cols) < 3:
        return
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(1 + .6 * len(cols), 1 + .55 * len(cols)))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                square=True, linewidths=.5, cbar_kws={"shrink": .7}, ax=ax)
    ax.set_title("Feature correlations, real data\n"
                 "Compare against fig1 from the synthetic dataset — where the "
                 "structure differs, your generator's assumptions did not hold",
                 fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}/figR4_correlation_real.png", dpi=200)
    plt.close(fig)
    print(f"  -> {out}/figR4_correlation_real.png")


def fig_load_season(df, out):
    if "session_load_au" not in df.columns:
        return
    d = df.groupby("date").agg(load=("session_load_au", "mean"),
                               rest=("is_rest_day", "mean"),
                               n=("player_id", "nunique"))
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    axes[0].plot(d.index, d["load"], lw=.9, color=PAL["load"], alpha=.55)
    axes[0].plot(d.index, d["load"].rolling(7, min_periods=1).mean(),
                 lw=2.2, color=PAL["risk"], label="7-day rolling mean")
    axes[0].set_ylabel("squad mean sRPE (AU)")
    axes[0].set_title("Training load across the season")
    axes[0].legend()
    axes[1].fill_between(d.index, d["rest"], color=PAL["ok"], alpha=.6)
    axes[1].set_ylabel("fraction\nresting")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("date")
    fig.tight_layout()
    fig.savefig(f"{out}/figR5_load_season.png", dpi=200)
    plt.close(fig)
    print(f"  -> {out}/figR5_load_season.png")


def fig_hooper_vs_load(df, out, n_players=4):
    if "hooper_index" not in df.columns:
        return
    label = dict(zip(df["player_id"], df.get("player_code", df["player_id"])))
    top = (df.groupby("player_id")["hooper_index"].count()
             .sort_values(ascending=False).head(n_players).index)
    fig, axes = plt.subplots(len(top), 1, figsize=(12, 2.6 * len(top)),
                             sharex=False)
    axes = np.atleast_1d(axes)
    for ax, pid in zip(axes, top):
        g = df[df["player_id"] == pid].sort_values("date")
        ax.plot(g["date"], g["hooper_index"], color=PAL["risk"], lw=1.3,
                label="Hooper index (higher = worse)")
        ax2 = ax.twinx()
        ax2.fill_between(g["date"], g["7_day_workload_average"], alpha=.25,
                         color=PAL["acc"], label="7-day mean load")
        ax2.grid(False)
        ax.set_ylabel("Hooper", fontsize=8)
        ax2.set_ylabel("load (AU)", fontsize=8)
        ax.set_title(f"player {label.get(pid, pid)}", fontsize=9, loc="left")
        ax.tick_params(labelsize=7)
        ax2.tick_params(labelsize=7)
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Subjective wellness against accumulated load, per player",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(f"{out}/figR6_hooper_vs_load.png", dpi=200)
    plt.close(fig)
    print(f"  -> {out}/figR6_hooper_vs_load.png")


def fig_wellness_trends(df, out):
    cols = _avail(df, WELLNESS)
    if not cols:
        return
    ncol = 2
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 2.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        g = df.groupby("date")[c].agg(["mean", "std"])
        g = g.rolling(7, min_periods=1).mean()
        ax.plot(g.index, g["mean"], color=PAL["acc"], lw=1.5)
        ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"],
                        color=PAL["acc"], alpha=.18)
        ax.set_title(f"{c}  (squad mean ±1 SD, 7-day smoothed)", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{out}/figR7_wellness_trends.png", dpi=200)
    plt.close(fig)
    print(f"  -> {out}/figR7_wellness_trends.png")


def fig_acwr(df, out):
    if "ACWR" not in df.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    a = df["ACWR"].dropna()
    axes[0].hist(a, bins=60, color=PAL["acc"], alpha=.75)
    for x, lab in ((0.8, "0.8"), (1.3, "1.3 — 'sweet spot' upper bound")):
        axes[0].axvline(x, ls="--", color=PAL["risk"], lw=1.3)
        axes[0].text(x, axes[0].get_ylim()[1] * .92, f" {lab}",
                     fontsize=8, color=PAL["risk"])
    axes[0].set_xlabel("ACWR")
    axes[0].set_ylabel("player-days")
    axes[0].set_title("ACWR distribution")

    if "y" in df.columns:
        d = df.dropna(subset=["ACWR", "y"]).copy()
        d["bin"] = pd.cut(d["ACWR"], [0, .8, 1.0, 1.3, 1.5, 10],
                          labels=["<0.8", "0.8–1.0", "1.0–1.3", "1.3–1.5", ">1.5"])
        r = d.groupby("bin", observed=True)["y"].agg(["mean", "count"])
        axes[1].bar(r.index.astype(str), r["mean"], color=PAL["risk"], alpha=.8)
        for i, (m, n) in enumerate(zip(r["mean"], r["count"])):
            axes[1].text(i, m, f"n={n:,}", ha="center", va="bottom", fontsize=7)
        axes[1].axhline(d["y"].mean(), ls="--", color="grey",
                        label=f"base rate {d['y'].mean():.2f}")
        axes[1].set_ylabel("next-day high-risk rate")
        axes[1].set_title("Does ACWR > 1.3 actually predict elevated risk here?")
        axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}/figR8_acwr.png", dpi=200)
    plt.close(fig)
    print(f"  -> {out}/figR8_acwr.png")


def fig_between_player(df, out):
    if "hooper_index" not in df.columns:
        return
    # short codes if the loader produced them; UUIDs are unreadable on an axis
    key = "player_code" if "player_code" in df.columns else "player_id"
    order = (df.groupby(key)["hooper_index"].median().sort_values().index)
    fig, ax = plt.subplots(figsize=(max(9, .32 * len(order)), 5))
    sns.boxplot(data=df, x=key, y="hooper_index", order=order,
                hue="team" if "team" in df.columns else None, dodge=False,
                palette={"TeamA": PAL["acc"], "TeamB": PAL["ok"]}
                if "team" in df.columns else None,
                color=PAL["acc"], fliersize=1.5, linewidth=.8, ax=ax)
    ax.set_title("Hooper index varies substantially between athletes\n"
                 "A single squad-wide threshold would mostly encode who "
                 "reports pessimistically — hence per-player thresholds",
                 fontsize=10)
    ax.set_xlabel("player")
    ax.set_ylabel("Hooper index (higher = worse)")
    plt.xticks(rotation=90, fontsize=8)
    if ax.get_legend():
        ax.legend(title="", fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{out}/figR9_between_player.png", dpi=200)
    plt.close(fig)
    print(f"  -> {out}/figR9_between_player.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(REAL_MODEL_CSV))
    ap.add_argument("--out", default=str(FIG_DIR / "real"))
    args = ap.parse_args()

    from pathlib import Path
    Path(args.out).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, parse_dates=["date"])
    print(f"{len(df):,} rows, {df['player_id'].nunique()} players, "
          f"{df['date'].min().date()} to {df['date'].max().date()}")
    print(f"available load features   : {_avail(df, LOAD)}")
    print(f"available wellness features: {_avail(df, WELLNESS)}\n")

    fig_correlation(df, args.out)
    fig_load_season(df, args.out)
    fig_hooper_vs_load(df, args.out)
    fig_wellness_trends(df, args.out)
    fig_acwr(df, args.out)
    fig_between_player(df, args.out)

    print("\nLook at these properly before filing them. Specifically: does "
          "ACWR > 1.3 associate with elevated risk in this squad (figR8), and "
          "does the correlation structure resemble your synthetic fig1 "
          "(figR4)? Where it does not, you have found Discussion material.")


if __name__ == "__main__":
    main()
