#!/usr/bin/env python3
"""
08_train_real.py — leakage-free training and validation on real data.

What is different from 03_model_training.py, and why it matters
---------------------------------------------------------------
1. ALL preprocessing (imputation, scaling) happens inside a sklearn
   Pipeline that is fitted on the training fold only. The original script
   fitted the imputer and scaler on the whole dataset before splitting,
   which leaks test-fold statistics into training and inflates every
   reported number.
2. StratifiedGroupKFold grouped on player_id is the primary within-sample
   scheme. Plain StratifiedKFold is retained only for comparability with the
   synthetic run and is labelled "optimistic" in the output.
3. The BiLSTM is validated with the SAME grouped/LOPO logic as the tree
   models. In the synthetic results the BiLSTM was evaluated on a temporal
   80/20 split and posted the best number -- an asymmetry that is not
   defensible in a viva.
4. PR-AUC and balanced accuracy are primary. With 10-20% positives, raw
   accuracy is uninformative (a model predicting "never at risk" scores 0.85).
5. Sequence windows never span a player boundary or a gap longer than
   LSTM_CONFIG["max_gap_days"].

Usage
-----
    python 08_train_real.py --models rf xgb --cv skf grouped lopo
    python 08_train_real.py --models lstm --cv lopo
    python 08_train_real.py --shap
    python 08_train_real.py --injury-check
    python 08_train_real.py --data football_athlete_monitoring.csv --label fatigue_state
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from config import (REAL_MODEL_CSV, REAL_FEATURES, GPS_FEATURES, RESULTS_DIR,
                    FIG_DIR, RANDOM_SEED, CV_CONFIG, LSTM_CONFIG)

warnings.filterwarnings("ignore")
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────
#  MODELS
# ─────────────────────────────────────────────────────────────

# Set by --fast. Shrinks the models so a full LOPO sweep finishes in
# seconds instead of minutes, for iterating on the pipeline. Always produce
# your final thesis numbers WITHOUT it.
FAST = False


def make_pipeline(kind: str, features: list[str]) -> Pipeline:
    """Impute -> scale -> classify, all fitted inside the fold."""
    pre = ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc", StandardScaler())]), features)],
        remainder="drop")

    if kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=60 if FAST else 300,
            max_depth=12 if FAST else None, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED)
    elif kind == "xgb":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed -- run `pip install xgboost`, "
                "or use --models rf") from exc
        clf = xgb.XGBClassifier(
            n_estimators=80 if FAST else 400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            eval_metric="logloss", n_jobs=-1, random_state=RANDOM_SEED)
    else:
        raise ValueError(kind)

    return Pipeline([("pre", pre), ("clf", clf)])


def scale_pos_weight(y: np.ndarray) -> float:
    pos = max(int(y.sum()), 1)
    return float((len(y) - pos) / pos)


# ─────────────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────────────

def metrics(y_true, y_pred, y_prob=None) -> dict:
    out = {
        "n":                 int(len(y_true)),
        "positives":         int(np.sum(y_true)),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "precision":         round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":            round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":                round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)
        out["pr_auc"]  = round(average_precision_score(y_true, y_prob), 4)
        # a PR-AUC below prevalence is worse than random guessing
        out["prevalence"] = round(float(np.mean(y_true)), 4)
        out["pr_auc_lift"] = round(out["pr_auc"] / out["prevalence"], 3) \
            if out["prevalence"] > 0 else None
    return out


def aggregate(fold_metrics: list[dict]) -> dict:
    if not fold_metrics:
        return {}
    keys = [k for k in fold_metrics[0]
            if isinstance(fold_metrics[0][k], (int, float))]
    agg = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if m.get(k) is not None]
        if vals:
            agg[k] = {"mean": round(float(np.mean(vals)), 4),
                      "std": round(float(np.std(vals)), 4)}
    agg["n_folds"] = len(fold_metrics)
    return agg


# ─────────────────────────────────────────────────────────────
#  CROSS-VALIDATION SCHEMES
# ─────────────────────────────────────────────────────────────

def _fold_cache(tag: str, kind: str, scheme: str) -> Path:
    d = RESULTS_DIR / "folds"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{tag}__{kind}__{scheme}.json"


def run_cv(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
           kind: str, scheme: str, features: list[str],
           tag: str = "run", resume: bool = True) -> dict:
    """
    Cross-validate with per-fold checkpointing.

    A full LOPO sweep is one model fit per player, which on a real dataset is
    minutes, not seconds. Completed folds are written to
    results/folds/<tag>__<model>__<scheme>.json as they finish, and skipped on
    a re-run. That makes a long sweep safe to interrupt, and safe to run in
    the background while you do something else.

    Delete the cache file (or pass --no-resume) to force a clean re-run.
    """
    cache = _fold_cache(tag, kind, scheme)
    done: dict[str, dict] = {}
    if resume and cache.exists():
        done = json.loads(cache.read_text())
        if done:
            print(f"      [resume] {len(done)} fold(s) already done in "
                  f"{cache.name}")

    if scheme == "skf":
        splitter = StratifiedKFold(CV_CONFIG["skf_splits"], shuffle=True,
                                   random_state=RANDOM_SEED)
        splits = list(enumerate(splitter.split(X, y)))

    elif scheme == "grouped":
        splitter = StratifiedGroupKFold(CV_CONFIG["grouped_splits"], shuffle=True,
                                        random_state=RANDOM_SEED)
        splits = list(enumerate(splitter.split(X, y, groups)))

    elif scheme == "lopo":
        # key folds by player so the cache stays valid if player order changes
        splits = [(p, (np.where(groups != p)[0], np.where(groups == p)[0]))
                  for p in np.unique(groups)]
    else:
        raise ValueError(scheme)

    skipped = 0
    for key, (tr, te) in splits:
        k = str(key)
        if k in done:
            continue
        if len(np.unique(y[tr])) < 2 or len(te) == 0:
            skipped += 1
            done[k] = {"__skipped__": True}
            cache.write_text(json.dumps(done))
            continue

        pipe = make_pipeline(kind, features)
        if kind == "xgb":
            pipe.named_steps["clf"].set_params(
                scale_pos_weight=scale_pos_weight(y[tr]))

        pipe.fit(X.iloc[tr], y[tr])
        pred = pipe.predict(X.iloc[te])
        prob = (pipe.predict_proba(X.iloc[te])[:, 1]
                if hasattr(pipe, "predict_proba") else None)

        # a held-out player with no positive days gives an undefined AUC;
        # keep the fold but let metrics() drop the AUC entries
        done[k] = metrics(y[te], pred, prob)
        cache.write_text(json.dumps(done))
        print(f"      fold {len(done)}/{len(splits)} done", flush=True)

    folds = [v for v in done.values() if not v.get("__skipped__")]
    skipped = sum(1 for v in done.values() if v.get("__skipped__"))

    agg = aggregate(folds)
    agg["skipped_folds"] = skipped
    agg["total_folds"] = len(splits)
    if skipped:
        print(f"      [note] {skipped} fold(s) skipped (single-class training "
              f"or empty test set). Report this number in Methods.")
    return agg


# ─────────────────────────────────────────────────────────────
#  BiLSTM  with player-grouped sequence splitting
# ─────────────────────────────────────────────────────────────

def build_sequences(df: pd.DataFrame, features: list[str], label_col: str,
                    seq_len: int, max_gap: int):
    """
    Returns X (n, seq_len, n_feat), y (n,), groups (n,).
    Windows never span a player boundary or a date gap > max_gap days.
    """
    Xs, ys, gs = [], [], []
    for pid, g in df.groupby("player_id", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        gaps = g["date"].diff().dt.days.fillna(1).to_numpy()
        vals = g[features].astype(float).to_numpy()
        labs = g[label_col].to_numpy()

        for i in range(seq_len, len(g)):
            window = slice(i - seq_len, i)
            if np.nanmax(gaps[window]) > max_gap:
                continue                      # window straddles a long absence
            if np.isnan(labs[i]):
                continue
            Xs.append(vals[window])
            ys.append(labs[i])
            gs.append(pid)

    if not Xs:
        return np.empty((0, seq_len, len(features))), np.array([]), np.array([])
    return np.asarray(Xs), np.asarray(ys, dtype=int), np.asarray(gs)


def run_lstm(df: pd.DataFrame, features: list[str], label_col: str,
             scheme: str = "grouped", tag: str = "run",
             resume: bool = True) -> dict:
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models, callbacks
    except ImportError:
        print("  [skip] TensorFlow not installed -- pip install tensorflow")
        return {}

    seq_len = LSTM_CONFIG["seq_len"]
    Xseq, yseq, gseq = build_sequences(df, features, label_col, seq_len,
                                       LSTM_CONFIG["max_gap_days"])
    print(f"    sequences: {len(Xseq):,}  positives: {int(yseq.sum()):,} "
          f"({yseq.mean():.1%})  players: {len(np.unique(gseq))}")
    if len(Xseq) < 200:
        print("  [skip] fewer than 200 usable sequences -- report this as a "
              "sample-size limitation rather than forcing a model.")
        return {}

    if scheme == "grouped":
        splitter = StratifiedGroupKFold(CV_CONFIG["grouped_splits"], shuffle=True,
                                        random_state=RANDOM_SEED)
        splits = list(enumerate(
            splitter.split(Xseq.reshape(len(Xseq), -1), yseq, gseq)))
    else:  # lopo
        splits = [(p, (np.where(gseq != p)[0], np.where(gseq == p)[0]))
                  for p in np.unique(gseq)]

    # Same per-fold checkpointing as the tree models. Training one LSTM per
    # player is slow, so an interrupted run must not lose completed folds.
    cache = _fold_cache(tag, "lstm", scheme)
    done: dict[str, dict] = {}
    if resume and cache.exists():
        done = json.loads(cache.read_text())
        if done:
            print(f"    [resume] {len(done)} fold(s) already done")

    for k, (tr, te) in splits:
        if str(k) in done:
            continue
        if len(np.unique(yseq[tr])) < 2 or len(te) == 0:
            done[str(k)] = {"__skipped__": True}
            cache.write_text(json.dumps(done))
            continue

        # scaler fitted on the TRAINING sequences only
        flat = Xseq[tr].reshape(-1, Xseq.shape[2])
        med = np.nanmedian(flat, axis=0)
        flat = np.where(np.isnan(flat), med, flat)
        mu, sd = flat.mean(0), flat.std(0) + 1e-8

        def prep(a):
            a = np.where(np.isnan(a), med, a)
            return (a - mu) / sd

        Xtr, Xte = prep(Xseq[tr]), prep(Xseq[te])

        model = models.Sequential([
            layers.Input(shape=(seq_len, Xseq.shape[2])),
            layers.Bidirectional(layers.LSTM(LSTM_CONFIG["units"],
                                             return_sequences=True)),
            layers.Dropout(LSTM_CONFIG["dropout"]),
            layers.Bidirectional(layers.LSTM(LSTM_CONFIG["units"] // 2)),
            layers.Dropout(LSTM_CONFIG["dropout"]),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy",
                      metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")])

        cw = None
        if LSTM_CONFIG["class_weight"]:
            classes = np.unique(yseq[tr])
            w = compute_class_weight("balanced", classes=classes, y=yseq[tr])
            cw = dict(zip(classes.tolist(), w.tolist()))

        model.fit(Xtr, yseq[tr],
                  validation_split=0.15,
                  epochs=LSTM_CONFIG["epochs"],
                  batch_size=LSTM_CONFIG["batch_size"],
                  class_weight=cw, verbose=0,
                  callbacks=[callbacks.EarlyStopping(
                      monitor="val_loss", patience=LSTM_CONFIG["patience"],
                      restore_best_weights=True)])

        prob = model.predict(Xte, verbose=0).ravel()
        done[str(k)] = metrics(yseq[te], (prob > 0.5).astype(int), prob)
        cache.write_text(json.dumps(done))
        print(f"      fold {len(done)}/{len(splits)}  "
              f"PR-AUC {done[str(k)].get('pr_auc', float('nan')):.3f}", flush=True)

    folds = [v for v in done.values() if not v.get("__skipped__")]
    agg = aggregate(folds)
    agg["total_folds"] = len(splits)
    return agg


# ─────────────────────────────────────────────────────────────
#  SHAP
# ─────────────────────────────────────────────────────────────

def tree_shap(clf, Xt: pd.DataFrame) -> np.ndarray:
    """
    Exact TreeSHAP values, robust across library versions.

    shap's TreeExplainer cannot parse XGBoost >= 3.0 model dumps (it raises
    on the bracketed `base_score` field). XGBoost computes the same exact
    TreeSHAP values natively via `pred_contribs=True`, so we use that for
    XGBoost and fall back to shap's explainer for scikit-learn forests.
    """
    if clf.__class__.__module__.startswith("xgboost"):
        import xgboost as xgb
        dm = xgb.DMatrix(Xt, feature_names=list(Xt.columns))
        contribs = clf.get_booster().predict(dm, pred_contribs=True)
        return contribs[:, :-1]          # drop the bias column

    import shap
    sv = shap.TreeExplainer(clf).shap_values(Xt)
    if isinstance(sv, list):
        sv = sv[1]
    if getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]
    return sv


def run_shap(X: pd.DataFrame, y: np.ndarray, features: list[str],
             out_dir: Path) -> None:
    try:
        import shap
    except ImportError:
        print("  [skip] shap not installed -- pip install shap")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    for kind in ("rf", "xgb"):
        print(f"  SHAP for {kind} …")
        try:
            pipe = make_pipeline(kind, features)
        except ImportError as exc:
            print(f"    [skip] {exc}")
            continue
        if kind == "xgb":
            pipe.named_steps["clf"].set_params(scale_pos_weight=scale_pos_weight(y))
        pipe.fit(X, y)

        Xt = pipe.named_steps["pre"].transform(X)
        Xt = pd.DataFrame(Xt, columns=features)
        sv = tree_shap(pipe.named_steps["clf"], Xt)

        for style, fn in (("bar", lambda: shap.summary_plot(
                               sv, Xt, plot_type="bar", show=False)),
                          ("beeswarm", lambda: shap.summary_plot(
                               sv, Xt, show=False))):
            plt.figure(figsize=(9, 7))
            fn()
            plt.tight_layout()
            p = out_dir / f"figR_shap_{kind}_{style}.png"
            plt.savefig(p, dpi=200)
            plt.close()
            print(f"    -> {p}")

        rank = (pd.Series(np.abs(sv).mean(0), index=features)
                  .sort_values(ascending=False))
        rank.to_csv(RESULTS_DIR / f"shap_ranking_real_{kind}.csv",
                    header=["mean_abs_shap"])
        print(f"    top 5: {list(rank.index[:5])}")

    print("\n  RQ3: compare these rankings against the synthetic ones with a "
          "Spearman rank correlation (09_transfer_analysis.py does this). "
          "Report the coefficient, not an impression.")


# ─────────────────────────────────────────────────────────────
#  RQ4 — do flagged days precede injury/illness?
# ─────────────────────────────────────────────────────────────

def injury_check(df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray,
                 features: list[str], windows=(3, 7)) -> None:
    if "injury_flag" not in df.columns and "illness_flag" not in df.columns:
        print("  [skip] no injury_flag / illness_flag column in the data.")
        return

    print("  Fitting a model on all data to score every player-day …")
    try:
        pipe = make_pipeline("xgb", features)
        pipe.named_steps["clf"].set_params(scale_pos_weight=scale_pos_weight(y))
    except ImportError:
        print("  [note] xgboost unavailable, falling back to random forest")
        pipe = make_pipeline("rf", features)
    pipe.fit(X, y)
    df = df.copy()
    df["risk"] = pipe.predict_proba(X)[:, 1]

    event = np.zeros(len(df), dtype=int)
    for c in ("injury_flag", "illness_flag"):
        if c in df.columns:
            event |= df[c].fillna(0).astype(int).to_numpy()
    df["event"] = event
    n_events = int(df["event"].sum())
    print(f"  {n_events} recorded injury/illness player-days")

    if n_events < 20:
        print("  [warn] Fewer than 20 events. This analysis is severely "
              "underpowered -- present it as exploratory, with a CI, and do "
              "not build an argument on it.")

    rows = []
    for w in windows:
        pre = np.zeros(len(df), dtype=bool)
        for _, g in df.groupby("player_id", sort=False):
            g = g.sort_values("date")
            ev = g["event"].to_numpy()
            flag = np.zeros(len(g), dtype=bool)
            for i in np.where(ev == 1)[0]:
                flag[max(0, i - w):i] = True
            pre[g.index] = flag

        pre_risk = df.loc[pre & (df["event"] == 0), "risk"]
        ctl_risk = df.loc[~pre & (df["event"] == 0), "risk"]
        if len(pre_risk) < 5 or len(ctl_risk) < 5:
            continue

        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(pre_risk, ctl_risk, alternative="greater")
        auc = u / (len(pre_risk) * len(ctl_risk))       # = probability of superiority
        rows.append({
            "window_days": w,
            "n_pre_event_days": len(pre_risk),
            "n_control_days": len(ctl_risk),
            "mean_risk_pre_event": round(pre_risk.mean(), 4),
            "mean_risk_control": round(ctl_risk.mean(), 4),
            "auc_prob_superiority": round(auc, 4),
            "mannwhitney_p": f"{p:.2e}",
        })

    if rows:
        res = pd.DataFrame(rows)
        print("\n" + res.to_string(index=False))
        res.to_csv(RESULTS_DIR / "injury_precedence.csv", index=False)
        print("\n  AUC of 0.5 means the model's risk score carries no "
              "information about impending injury. Above ~0.6 is worth "
              "discussing; report it with the caveat about power.")


# ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(REAL_MODEL_CSV))
    ap.add_argument("--label", default="y")
    ap.add_argument("--models", nargs="+", default=["rf", "xgb"],
                    choices=["rf", "xgb", "lstm"])
    ap.add_argument("--cv", nargs="+", default=["grouped", "lopo"],
                    choices=["skf", "grouped", "lopo"])
    ap.add_argument("--include-gps", action="store_true",
                    help="add GPS features (Track B only, after 10_gps_aggregate.py)")
    ap.add_argument("--shap", action="store_true")
    ap.add_argument("--injury-check", action="store_true")
    ap.add_argument("--tag", default="real", help="suffix for output files")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore cached folds and re-run from scratch")
    ap.add_argument("--fast", action="store_true",
                    help="smaller models for quick iteration -- NOT for final numbers")
    args = ap.parse_args()

    if args.fast:
        global FAST
        FAST = True
        print("[--fast] reduced model size; do not report these numbers.\n")

    df = pd.read_csv(args.data, parse_dates=["date"]).sort_values(
        ["player_id", "date"]).reset_index(drop=True)

    features = [f for f in REAL_FEATURES if f in df.columns]
    if args.include_gps:
        features += [f for f in GPS_FEATURES if f in df.columns]
    missing = [f for f in REAL_FEATURES if f not in df.columns]

    if args.label not in df.columns:
        raise SystemExit(f"Label column '{args.label}' not found. "
                         f"Run 07_labels.py first.")

    df = df.dropna(subset=[args.label]).reset_index(drop=True)
    X = df[features]
    y = df[args.label].astype(int).to_numpy()
    groups = df["player_id"].to_numpy()

    print("=" * 72)
    print(f"DATA   {args.data}")
    print(f"  {len(df):,} labelled rows, {len(np.unique(groups))} players")
    print(f"  label '{args.label}': {y.sum():,} positives ({y.mean():.1%})")
    print(f"  {len(features)} features used")
    if missing:
        print(f"  NOT AVAILABLE ({len(missing)}): {missing}")
        print("  -> list these in your Limitations section")
    print("=" * 72)

    results = {}
    for kind in args.models:
        if kind == "lstm":
            for scheme in [s for s in args.cv if s in ("grouped", "lopo")]:
                print(f"\n[lstm / {scheme}]")
                results.setdefault("lstm", {})[scheme] = run_lstm(
                    df, features, args.label, scheme,
                    tag=args.tag, resume=not args.no_resume)
            continue

        for scheme in args.cv:
            print(f"\n[{kind} / {scheme}]"
                  + ("   <- OPTIMISTIC: same players in train and test"
                     if scheme == "skf" else ""))
            r = run_cv(X, y, groups, kind, scheme, features,
                       tag=args.tag, resume=not args.no_resume)
            results.setdefault(kind, {})[scheme] = r
            for m in ("pr_auc", "balanced_accuracy", "f1", "roc_auc"):
                if m in r:
                    print(f"      {m:18s} {r[m]['mean']:.4f} ± {r[m]['std']:.4f}")

    if results:
        out = RESULTS_DIR / f"metrics_{args.tag}.json"
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nMetrics -> {out}")

        if "lopo" in args.cv and any("lopo" in v for v in results.values()):
            print("\nReport LOPO as your headline metric. If LOPO PR-AUC is "
                  "close to prevalence, the model does not generalise to new "
                  "players -- which is a legitimate and important finding, not "
                  "a failure to be tuned away.")

    if args.shap:
        print("\n[SHAP]")
        run_shap(X, y, features, FIG_DIR / args.tag)

    if args.injury_check:
        print("\n[RQ4 injury precedence]")
        injury_check(df, X, y, features)


if __name__ == "__main__":
    main()
