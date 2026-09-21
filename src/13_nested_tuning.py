#!/usr/bin/env python3
"""
13_nested_tuning.py — nested cross-validation for the paper.

The thesis reported untuned models on purpose, to avoid selection bias.
A reviewer will ask whether the conclusion survives tuning. This answers that
without cheating: the hyperparameter search runs *inside* each training fold,
so the outer estimate never sees the data it is scored on.

    outer:  StratifiedGroupKFold(5) on player_id   -> honest estimate
    inner:  StratifiedGroupKFold(3) on the training part only -> model choice

Run:
    python 13_nested_tuning.py                 # both models, all four blocks
    python 13_nested_tuning.py --models rf     # one family
    python 13_nested_tuning.py --quick         # small grid, for a smoke test

Writes results/nested_tuned.json and prints a comparison against the
untuned numbers already in results/incremental_value_rf.json.
"""
from __future__ import annotations

import argparse
import json
import warnings
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_IV = import_module("11_incremental_value")
from config import RESULTS_DIR, RANDOM_SEED  # noqa: E402

BLOCKS = _IV.BLOCKS


# ─────────────────────────────────────────────────────────────
#  SEARCH SPACES
# ─────────────────────────────────────────────────────────────

def grid(kind: str, quick: bool) -> dict:
    if kind == "rf":
        if quick:
            return {"clf__max_depth": [None, 8],
                    "clf__min_samples_leaf": [1, 20]}
        return {
            "clf__n_estimators":     [300],
            "clf__max_depth":        [None, 12],
            "clf__min_samples_leaf": [1, 5, 20],
        }
    if quick:
        return {"clf__max_depth": [3, 6], "clf__learning_rate": [0.05]}
    return {
        "clf__n_estimators":     [400],
        "clf__max_depth":        [3, 4, 6],
        "clf__learning_rate":    [0.05, 0.10],
        "clf__reg_lambda":       [1.0, 5.0],
    }


def base_pipeline(kind: str, feats: list[str], pos_weight: float | None):
    pre = ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc", StandardScaler())]), feats)],
        remainder="drop")
    if kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_SEED)
    else:
        import xgboost as xgb
        clf = xgb.XGBClassifier(
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            n_jobs=-1, random_state=RANDOM_SEED,
            scale_pos_weight=pos_weight if pos_weight else 1.0)
    return Pipeline([("pre", pre), ("clf", clf)])


# ─────────────────────────────────────────────────────────────
#  NESTED EVALUATION
# ─────────────────────────────────────────────────────────────

CACHE = RESULTS_DIR / "tuning_cache"


def _cache(kind, block, fold):
    CACHE.mkdir(parents=True, exist_ok=True)
    safe = block.replace(" ", "_").replace("+", "plus")
    return CACHE / f"{kind}__{safe}__f{fold}.json"


def nested_evaluate(df, feats, label, kind, quick=False, seed=RANDOM_SEED,
                    block=""):
    """Outer folds give the estimate. Inner folds choose the model.

    Each outer fold is cached, so an interrupted run resumes where it stopped.
    """
    feats = [f for f in feats if f in df.columns]
    X, y = df[feats], df[label].astype(int).to_numpy()
    groups = df["player_id"].to_numpy()

    outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    fold_pr, fold_roc, chosen = [], [], []

    for k, (tr, te) in enumerate(outer.split(X, y, groups), 1):
        if len(np.unique(y[tr])) < 2:
            continue
        cf = _cache(kind, block, k)
        if cf.exists():
            c = json.loads(cf.read_text())
            oof[te] = np.array(c["pred"])
            fold_pr.append(c["pr"]); fold_roc.append(c["roc"])
            chosen.append(c["params"])
            print(f"      fold {k}: cached  PR-AUC {c['pr']:.4f}", flush=True)
            continue

        pos = float((len(y[tr]) - y[tr].sum()) / max(y[tr].sum(), 1))
        pipe = base_pipeline(kind, feats, pos)

        inner = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
        search = GridSearchCV(
            pipe, grid(kind, quick), scoring="average_precision",
            cv=list(inner.split(X.iloc[tr], y[tr], groups[tr])),
            n_jobs=-1, refit=True, error_score="raise")
        search.fit(X.iloc[tr], y[tr])

        p = search.best_estimator_.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        pr = float(average_precision_score(y[te], p))
        rc = float(roc_auc_score(y[te], p))
        fold_pr.append(pr); fold_roc.append(rc)
        params = {kk.replace("clf__", ""): vv
                  for kk, vv in search.best_params_.items()}
        chosen.append(params)
        cf.write_text(json.dumps(
            {"pred": p.tolist(), "pr": pr, "roc": rc, "params": params}))
        print(f"      fold {k}: PR-AUC {pr:.4f}  params {params}", flush=True)

    return {
        "n_features":  len(feats),
        "pr_auc":      float(np.mean(fold_pr)),
        "pr_auc_sd":   float(np.std(fold_pr)),
        "roc_auc":     float(np.mean(fold_roc)),
        "roc_auc_sd":  float(np.std(fold_roc)),
        "prevalence":  float(y.mean()),
        "chosen_per_fold": chosen,
    }, oof, y, groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["rf", "xgb"])
    ap.add_argument("--quick", action="store_true",
                    help="small grid, for checking the script runs")
    ap.add_argument("--label", default="y")
    ap.add_argument("--data", default="data/real/soccermon_gps_subset.csv")
    args = ap.parse_args()

    path = Path(args.data)
    df = pd.read_csv(path)
    df = df.dropna(subset=[args.label])
    print(f"data: {path.name}   rows {len(df):,}   "
          f"players {df.player_id.nunique()}   "
          f"prevalence {df[args.label].mean():.3f}\n")

    out = {"label": args.label, "quick": args.quick, "models": {}}

    for kind in args.models:
        print(f"=== {kind.upper()} ===")
        blocks, oofs = {}, {}
        for name, feats in BLOCKS:
            print(f"  {name}")
            res, oof, y, groups = nested_evaluate(
                df, feats, args.label, kind, args.quick, block=name)
            blocks[name] = res
            oofs[name] = oof
            print(f"    -> PR-AUC {res['pr_auc']:.4f} +/- {res['pr_auc_sd']:.4f}"
                  f"   ROC-AUC {res['roc_auc']:.4f}\n", flush=True)

        # bootstrap the incremental steps, resampling players not rows
        names = [n for n, _ in BLOCKS]
        deltas = {}
        for a, b in zip(names, names[1:]):
            d = _IV.bootstrap_delta(oofs[a], oofs[b], y, groups)
            deltas[f"{a} -> {b}"] = d
            pr=d['delta_pr_auc']
            print(f"  {a} -> {b}:  dPR-AUC {pr['mean']:+.4f} "
                  f"95% CI [{pr['lo']:+.4f}, {pr['hi']:+.4f}]  "
                  f"P(>0)={pr['p_gt_0']:.3f}")
        d = _IV.bootstrap_delta(oofs[names[0]], oofs[names[-1]], y, groups)
        deltas[f"{names[0]} -> {names[-1]}"] = d
        print()
        out["models"][kind] = {"blocks": blocks, "deltas": deltas}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = RESULTS_DIR / "nested_tuned.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"written: {dest}")

    # side by side with the untuned thesis numbers
    old = RESULTS_DIR / "incremental_value_rf.json"
    if old.exists() and "rf" in out["models"]:
        o = json.loads(old.read_text())["blocks"]
        n = out["models"]["rf"]["blocks"]
        print("\nUNTUNED (thesis) vs TUNED (paper), Random Forest")
        print(f"{'block':22s} {'untuned PR':>11s} {'tuned PR':>10s} "
              f"{'untuned ROC':>12s} {'tuned ROC':>10s}")
        for k in n:
            if k in o:
                print(f"{k:22s} {o[k]['pr_auc']:11.4f} {n[k]['pr_auc']:10.4f} "
                      f"{o[k]['roc_auc']:12.4f} {n[k]['roc_auc']:10.4f}")


if __name__ == "__main__":
    main()
