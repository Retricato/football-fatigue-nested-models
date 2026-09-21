#!/usr/bin/env python3
"""
07_labels.py — construct fatigue-risk labels and run the sensitivity analysis.

This is the most contestable step in the thesis. Synthetic labels came from
generator rules; real labels must be derived from observed data, and any
derivation embeds assumptions. The response is not to find the "right"
threshold but to define several, report all of them, and be explicit about
which one is primary and why.

Labels implemented
------------------
  A_hooper_1.0sd      primary   Hooper Index > player's trailing mean + 1.0 SD
  B_readiness_tertile secondary readiness in the player's bottom tertile
  C_hooper_1.5sd      sensitivity, stricter
  D_hooper_0.5sd      sensitivity, looser

Every label is shifted forward by LABEL_HORIZON_DAYS (default 1), so features
observed on day t predict risk on day t+1. Same-day prediction would make the
wellness features near-tautological and would be useless to a coach.

Usage
-----
    python 07_labels.py
    python 07_labels.py --data data/real/soccermon_cohort.csv
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (REAL_DAILY_CSV, REAL_MODEL_CSV, LABEL_CONFIGS,
                    PRIMARY_LABEL, LABEL_HORIZON_DAYS, FIG_DIR, RESULTS_DIR)


# ─────────────────────────────────────────────────────────────
#  LABEL CONSTRUCTORS
# ─────────────────────────────────────────────────────────────

def rolling_z_label(g: pd.DataFrame, source: str, window: int,
                    threshold: float, min_periods: int) -> pd.Series:
    """
    High risk when today's value exceeds this player's own trailing mean by
    more than `threshold` SDs.

    Per-player thresholding is not a stylistic choice. Hooper scores are not
    comparable between athletes -- one player's habitual 12 is another's
    crisis (Hooper & Mackinnon, 1995). A squad-wide cutoff would mostly
    encode who reports pessimistically.

    Windows are TRAILING and shifted by one day, so no same-day or future
    information enters the baseline.
    """
    s = g[source]
    base = s.shift(1).rolling(window, min_periods=min_periods)
    mu, sd = base.mean(), base.std()
    z = (s - mu) / sd.replace(0, np.nan)
    lab = (z > threshold).astype("float")
    lab[z.isna()] = np.nan          # undefined until the baseline exists
    return lab


def within_player_quantile_label(g: pd.DataFrame, source: str,
                                 quantile: float, direction: str) -> pd.Series:
    """
    Bottom (or top) quantile of a variable within a player's own record.

    Note the trade-off: this uses the player's full-season distribution, so it
    is not deployable prospectively. It is fine as a secondary, descriptive
    label for comparison with Label A, but say so in Methods -- do not present
    it as an operational rule.
    """
    s = g[source]
    if s.notna().sum() < 20:
        return pd.Series(np.nan, index=g.index)
    cut = s.quantile(quantile)
    lab = (s < cut) if direction == "below" else (s > cut)
    out = lab.astype("float")
    out[s.isna()] = np.nan
    return out


def build_label(df: pd.DataFrame, name: str, cfg: dict,
                horizon: int = LABEL_HORIZON_DAYS) -> pd.Series:
    src = cfg["source"]
    if src not in df.columns:
        print(f"  [skip] {name}: source column '{src}' not in data "
              f"(did you run 06_build_real_dataset.py --features?)")
        return pd.Series(np.nan, index=df.index)

    parts = []
    for _, g in df.groupby("player_id", sort=False):
        g = g.sort_values("date")
        if cfg["method"] == "rolling_z":
            lab = rolling_z_label(g, src, cfg["window"], cfg["threshold"],
                                  cfg["min_periods"])
        elif cfg["method"] == "within_player_quantile":
            lab = within_player_quantile_label(g, src, cfg["quantile"],
                                               cfg["direction"])
        else:
            raise ValueError(cfg["method"])

        # shift so features at t predict the label at t+horizon
        parts.append(lab.shift(-horizon))

    return pd.concat(parts).reindex(df.index)


# ─────────────────────────────────────────────────────────────
#  REPORTING
# ─────────────────────────────────────────────────────────────

def prevalence_table(df: pd.DataFrame, label_cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in label_cols:
        s = df[c]
        defined = s.notna().sum()
        pos = int(s.sum()) if defined else 0
        per_player = (df.dropna(subset=[c]).groupby("player_id")[c]
                        .agg(["sum", "count"]))
        rows.append({
            "label": c,
            "defined_rows": int(defined),
            "positives": pos,
            "prevalence": round(pos / defined, 4) if defined else np.nan,
            "players_with_0_positives": int((per_player["sum"] == 0).sum()),
            "min_player_prevalence": round(
                (per_player["sum"] / per_player["count"]).min(), 3)
            if len(per_player) else np.nan,
            "max_player_prevalence": round(
                (per_player["sum"] / per_player["count"]).max(), 3)
            if len(per_player) else np.nan,
        })
    return pd.DataFrame(rows)


def agreement_matrix(df: pd.DataFrame, label_cols: list[str]) -> pd.DataFrame:
    """Cohen's kappa between label definitions."""
    from sklearn.metrics import cohen_kappa_score
    m = pd.DataFrame(index=label_cols, columns=label_cols, dtype=float)
    for a in label_cols:
        for b in label_cols:
            if a == b:
                m.loc[a, b] = 1.0
                continue
            sub = df.loc[:, [a, b]].dropna()
            m.loc[a, b] = (round(cohen_kappa_score(sub[a].astype(int),
                                                   sub[b].astype(int)), 3)
                           if len(sub) > 30 else np.nan)
    return m


def plot_label_distributions(df: pd.DataFrame, label_cols: list[str],
                             out: str) -> None:
    per = []
    for c in label_cols:
        g = df.dropna(subset=[c]).groupby("player_id")[c].mean()
        for pid, v in g.items():
            per.append({"label": c, "player_id": pid, "prevalence": v})
    if not per:
        return
    pdf = pd.DataFrame(per)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, c in enumerate(label_cols):
        vals = pdf[pdf["label"] == c]["prevalence"]
        ax.scatter(np.full(len(vals), i) + np.random.uniform(-.13, .13, len(vals)),
                   vals, alpha=.65, s=26)
    ax.set_xticks(range(len(label_cols)))
    ax.set_xticklabels(label_cols, rotation=20, ha="right")
    ax.set_ylabel("per-player positive rate")
    ax.set_title("Label prevalence varies substantially between players\n"
                 "(each point is one athlete)")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  figure -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(REAL_DAILY_CSV))
    ap.add_argument("--out", default=str(REAL_MODEL_CSV))
    ap.add_argument("--horizon", type=int, default=LABEL_HORIZON_DAYS)
    args = ap.parse_args()

    df = pd.read_csv(args.data, parse_dates=["date"]).sort_values(
        ["player_id", "date"]).reset_index(drop=True)
    print(f"Loaded {len(df):,} rows, {df['player_id'].nunique()} players")
    print(f"Prediction horizon: t -> t+{args.horizon} day(s)\n")

    print("BUILDING LABELS")
    label_cols = []
    for name, cfg in LABEL_CONFIGS.items():
        col = f"label_{name}"
        df[col] = build_label(df, name, cfg, horizon=args.horizon)
        if df[col].notna().any():
            label_cols.append(col)
            print(f"  {col:28s} {int(df[col].sum()):>6,} positives / "
                  f"{int(df[col].notna().sum()):>6,} defined")

    if not label_cols:
        raise SystemExit(
            "\nNo labels could be built. Run 06_build_real_dataset.py --features "
            "first so hooper_index and readiness exist.")

    primary = f"label_{PRIMARY_LABEL}"
    if primary in df.columns:
        df["y"] = df[primary]
        print(f"\n  primary label -> 'y'  ({primary})")

    print("\nPREVALENCE")
    prev = prevalence_table(df, label_cols)
    print(prev.to_string(index=False))
    prev.to_csv(RESULTS_DIR / "label_prevalence.csv", index=False)

    if (prev["prevalence"] < 0.05).any():
        print("\n  [WARN] A label has under 5% positives. LOPO folds will be "
              "unstable. Consider the 0.5 SD variant as primary and say why.")
    if (prev["players_with_0_positives"] > 0).any():
        print("  [note] Some players have zero positive days. 08_train_real.py "
              "skips them in LOPO and reports how many -- state that number.")

    print("\nAGREEMENT BETWEEN LABEL DEFINITIONS (Cohen's kappa)")
    kap = agreement_matrix(df, label_cols)
    print(kap.to_string())
    kap.to_csv(RESULTS_DIR / "label_agreement.csv")
    print("\n  Kappa between the Hooper label and the readiness label is worth a "
          "paragraph in Discussion: they measure related but distinct constructs, "
          "and low agreement means 'fatigue risk' is not a single well-defined "
          "target -- which is itself a finding.")

    FIG_DIR.joinpath("real").mkdir(parents=True, exist_ok=True)
    plot_label_distributions(df, label_cols,
                             str(FIG_DIR / "real" / "figR3_label_prevalence.png"))

    df.to_csv(args.out, index=False)
    print(f"\nWROTE {args.out}   ({len(df):,} rows x {df.shape[1]} cols)")
    print("Next: python 08_train_real.py --models rf xgb --cv skf grouped lopo")


if __name__ == "__main__":
    main()
