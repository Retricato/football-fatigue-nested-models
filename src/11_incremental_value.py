#!/usr/bin/env python3
"""
11_incremental_value.py — nested feature blocks + bootstrap CIs.

Two analyses that a reviewer will demand and that the thesis does not yet have.

1. INCREMENTAL VALUE OVER PERSISTENCE
   The label is derived from the wellness questionnaire, and yesterday's
   questionnaire is also an input. So part of any model's performance is
   simple autocorrelation -- "she felt bad yesterday, she'll feel bad today".

   This fits four nested models on IDENTICAL rows and folds:

       M0  persistence     today's Hooper index alone (1 feature)
       M1  + wellness      all self-reported items
       M2  + training load sRPE, ACWR, rolling load, monotony, strain
       M3  + GPS           objective external load

   The gap M3 - M0 is what the modelling actually buys over doing nothing.
   Reporting M0 pre-empts the strongest objection to the whole paper.

2. BOOTSTRAP CONFIDENCE INTERVALS
   Resampled over PLAYERS, not rows, because rows within a player are not
   independent. Four consistent comparisons is suggestive; a CI excluding
   zero is evidence.

Usage
-----
    python 11_incremental_value.py
    python 11_incremental_value.py --model xgb --boot 1000
"""

import argparse
import json
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from importlib import import_module

_T = import_module("08_train_real")
from config import (REAL_FEATURES, GPS_FEATURES, RESULTS_DIR, RANDOM_SEED,
                    CV_CONFIG)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  NESTED FEATURE BLOCKS
# ─────────────────────────────────────────────────────────────

PERSISTENCE = ["hooper_index"]

WELLNESS = ["hooper_index", "sleep_quality", "muscle_soreness", "mood_score",
            "fatigue_score", "stress_score", "sleep_hours", "readiness",
            "7_day_hooper_trend"]

LOAD = ["RPE", "session_duration_minutes", "session_load_au", "n_sessions",
        "ACWR", "7_day_workload_average", "28_day_workload_average",
        "cumulative_fatigue_score", "monotony", "strain", "days_since_rest",
        "acwr_provided", "atl_provided", "ctl28_provided", "monotony_provided"]

BLOCKS = [
    ("M0  persistence",    PERSISTENCE),
    ("M1  + wellness",     WELLNESS),
    ("M2  + training load", WELLNESS + LOAD),
    ("M3  + GPS",          WELLNESS + LOAD + GPS_FEATURES),
]


def evaluate_block(df, feats, label, kind, seed=RANDOM_SEED):
    """Grouped CV; returns per-fold (pr_auc, roc_auc) and pooled predictions."""
    feats = [f for f in feats if f in df.columns]
    X, y = df[feats], df[label].astype(int).to_numpy()
    groups = df["player_id"].to_numpy()

    splitter = StratifiedGroupKFold(CV_CONFIG["grouped_splits"], shuffle=True,
                                    random_state=seed)
    oof = np.full(len(y), np.nan)
    fold_pr, fold_roc = [], []

    for tr, te in splitter.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        pipe = _T.make_pipeline(kind, feats)
        if kind == "xgb":
            pipe.named_steps["clf"].set_params(
                scale_pos_weight=_T.scale_pos_weight(y[tr]))
        pipe.fit(X.iloc[tr], y[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        if len(np.unique(y[te])) > 1:
            fold_pr.append(average_precision_score(y[te], p))
            fold_roc.append(roc_auc_score(y[te], p))

    return {
        "n_features": len(feats),
        "pr_auc": float(np.mean(fold_pr)),
        "pr_auc_sd": float(np.std(fold_pr)),
        "roc_auc": float(np.mean(fold_roc)),
        "roc_auc_sd": float(np.std(fold_roc)),
        "prevalence": float(y.mean()),
    }, oof, y, groups


def bootstrap_delta(oof_a, oof_b, y, groups, n_boot=1000, seed=RANDOM_SEED):
    """
    Paired bootstrap over PLAYERS of the difference in PR-AUC and ROC-AUC.

    Clustering on player matters: consecutive days from one athlete are
    highly dependent, so a row-level bootstrap would give CIs that are far
    too narrow.
    """
    rng = np.random.default_rng(seed)
    players = np.unique(groups)
    d_pr, d_roc = [], []

    for _ in range(n_boot):
        pick = rng.choice(players, size=len(players), replace=True)
        idx = np.concatenate([np.where(groups == p)[0] for p in pick])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        a, b = oof_a[idx], oof_b[idx]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 50 or len(np.unique(yy[ok])) < 2:
            continue
        d_pr.append(average_precision_score(yy[ok], b[ok])
                    - average_precision_score(yy[ok], a[ok]))
        d_roc.append(roc_auc_score(yy[ok], b[ok])
                     - roc_auc_score(yy[ok], a[ok]))

    def ci(v):
        v = np.asarray(v)
        return {"mean": float(v.mean()),
                "lo": float(np.percentile(v, 2.5)),
                "hi": float(np.percentile(v, 97.5)),
                "p_gt_0": float((v > 0).mean())}

    return {"delta_pr_auc": ci(d_pr), "delta_roc_auc": ci(d_roc),
            "n_boot": len(d_pr)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/real/soccermon_gps_subset.csv")
    ap.add_argument("--label", default="y")
    ap.add_argument("--model", default="rf", choices=["rf", "xgb"])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    if args.fast:
        _T.FAST = True
        print("[--fast] reduced model size -- not for final numbers\n")

    df = pd.read_csv(args.data).dropna(subset=[args.label])
    print(f"{args.data}")
    print(f"  {len(df):,} rows, {df['player_id'].nunique()} players, "
          f"{df[args.label].mean():.1%} positive\n")

    results, oofs = {}, {}
    print(f"{'model':22s} {'feats':>6s} {'PR-AUC':>16s} {'ROC-AUC':>16s}")
    print("-" * 64)
    for name, feats in BLOCKS:
        r, oof, y, groups = evaluate_block(df, feats, args.label, args.model)
        results[name], oofs[name] = r, oof
        print(f"{name:22s} {r['n_features']:>6d} "
              f"{r['pr_auc']:.3f} ± {r['pr_auc_sd']:.3f}   "
              f"{r['roc_auc']:.3f} ± {r['roc_auc_sd']:.3f}")

    base = results["M0  persistence"]
    full = results["M3  + GPS"]
    print(f"\n  prevalence {base['prevalence']:.3f}")
    print(f"  M0 -> M3 gain: PR-AUC {full['pr_auc'] - base['pr_auc']:+.3f}, "
          f"ROC-AUC {full['roc_auc'] - base['roc_auc']:+.3f}")

    print(f"\nBOOTSTRAP over players ({args.boot} resamples)")
    cis = {}
    for a, b in (("M0  persistence", "M3  + GPS"),
                 ("M2  + training load", "M3  + GPS"),
                 ("M0  persistence", "M2  + training load")):
        c = bootstrap_delta(oofs[a], oofs[b], y, groups, args.boot)
        cis[f"{a} -> {b}"] = c
        pr, roc = c["delta_pr_auc"], c["delta_roc_auc"]
        star = "  *" if pr["lo"] > 0 else ""
        print(f"  {a} -> {b}")
        print(f"     dPR-AUC  {pr['mean']:+.3f}  95% CI [{pr['lo']:+.3f}, {pr['hi']:+.3f}]{star}")
        print(f"     dROC-AUC {roc['mean']:+.3f}  95% CI [{roc['lo']:+.3f}, {roc['hi']:+.3f}]")

    out = RESULTS_DIR / f"incremental_value_{args.model}.json"
    out.write_text(json.dumps({"blocks": results, "bootstrap": cis}, indent=2))
    print(f"\nSaved -> {out}")
    print("\n  A CI on the M2 -> M3 step that excludes zero is the evidence "
          "that GPS adds value. The M0 row is what stops a reviewer claiming "
          "your model is just autocorrelation.")


if __name__ == "__main__":
    main()
