#!/usr/bin/env python3
"""
09_transfer_analysis.py — synthetic <-> real transfer analysis.

This is the novel contribution of the thesis. Three experiments:

  1. DISTRIBUTION MATCH
     Two-sample KS test per shared feature, with effect sizes and overlaid
     densities. Note: with n in the thousands, KS rejects almost everything,
     so lead with the effect sizes and the plots. The pattern of divergence
     is the interesting part, not the p-values.

  2. CROSS-DOMAIN TRANSFER
     Train on synthetic, test on real; and the reverse. The performance drop
     quantifies how realistic the generator is -- a single interpretable
     number that expresses the contribution better than any table.

  3. SHAP RANK AGREEMENT (RQ3)
     Spearman correlation between the synthetic and real feature-importance
     orderings. Agreement supports the generator's construct validity;
     disagreement identifies which encoded sports-science assumptions the
     real data does not support. Both are results.

Usage
-----
    python 09_transfer_analysis.py
    python 09_transfer_analysis.py --finetune      # optional BiLSTM experiment
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)

from config import (SYNTHETIC_CSV, REAL_MODEL_CSV, TRANSFER_FEATURES,
                    RESULTS_DIR, FIG_DIR, RANDOM_SEED)

import sys
sys.path.insert(0, str(RESULTS_DIR.parent))
from importlib import import_module
_train = import_module("08_train_real")
make_pipeline, scale_pos_weight = _train.make_pipeline, _train.scale_pos_weight


# ─────────────────────────────────────────────────────────────
#  LOADING — put both domains on a common footing
# ─────────────────────────────────────────────────────────────

def load_synthetic(path, features):
    df = pd.read_csv(path)
    # synthetic label: fatigue_state is Low/Moderate/High -> binary "High"
    if "fatigue_state" in df.columns:
        df["y"] = (df["fatigue_state"].astype(str)
                     .str.strip().str.lower().eq("high")).astype(int)
    present = [f for f in features if f in df.columns]
    return df, present


def load_real(path, features):
    df = pd.read_csv(path, parse_dates=["date"])
    if "y" not in df.columns:
        raise SystemExit("Real data has no 'y' column -- run 07_labels.py.")
    df = df.dropna(subset=["y"])
    df["y"] = df["y"].astype(int)
    present = [f for f in features if f in df.columns]
    return df, present


# ─────────────────────────────────────────────────────────────
#  1. DISTRIBUTION MATCH
# ─────────────────────────────────────────────────────────────

def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s) if s else np.nan


def distribution_match(syn, real, features):
    rows = []
    for f in features:
        a = syn[f].dropna().to_numpy(float)
        b = real[f].dropna().to_numpy(float)
        if len(a) < 20 or len(b) < 20:
            continue
        ks, p = ks_2samp(a, b)
        rows.append({
            "feature": f,
            "synth_mean": round(a.mean(), 3), "real_mean": round(b.mean(), 3),
            "synth_sd": round(a.std(), 3),    "real_sd": round(b.std(), 3),
            "ks_statistic": round(ks, 4),
            "ks_p": p,
            "cohens_d": round(cohens_d(a, b), 3),
        })

    res = pd.DataFrame(rows)
    if res.empty:
        return res

    # Holm correction across features
    order = res["ks_p"].rank(method="first").astype(int)
    m = len(res)
    res["p_holm"] = np.minimum(1.0, res["ks_p"] * (m - order + 1))
    res["differs"] = res["p_holm"] < 0.05
    res = res.sort_values("ks_statistic", ascending=False)
    res["ks_p"] = res["ks_p"].map(lambda x: f"{x:.2e}")
    res["p_holm"] = res["p_holm"].map(lambda x: f"{x:.2e}")
    return res


def plot_overlays(syn, real, features, out):
    feats = [f for f in features if f in syn.columns and f in real.columns][:9]
    if not feats:
        return
    n = len(feats)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.1 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, feats):
        for d, lab, c in ((syn, "synthetic", "#3498db"), (real, "real", "#e74c3c")):
            v = pd.to_numeric(d[f], errors="coerce").dropna()
            if len(v) > 10:
                ax.hist(v, bins=40, density=True, alpha=.5, label=lab, color=c)
        ax.set_title(f, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[len(feats):]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    fig.suptitle("Synthetic vs. real feature distributions "
                 "(shared feature subset)", y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  figure -> {out}")


# ─────────────────────────────────────────────────────────────
#  2. CROSS-DOMAIN TRANSFER
# ─────────────────────────────────────────────────────────────

def evaluate(pipe, X, y):
    pred = pipe.predict(X)
    prob = pipe.predict_proba(X)[:, 1]
    out = {
        "n": int(len(y)),
        "prevalence": round(float(np.mean(y)), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y, pred), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
    }
    if len(np.unique(y)) > 1:
        out["roc_auc"] = round(roc_auc_score(y, prob), 4)
        out["pr_auc"] = round(average_precision_score(y, prob), 4)
        out["pr_auc_lift"] = round(out["pr_auc"] / out["prevalence"], 3)
    return out


def _fit(kind, features, X, y):
    p = make_pipeline(kind, features)
    if kind == "xgb":
        p.named_steps["clf"].set_params(scale_pos_weight=scale_pos_weight(y))
    p.fit(X, y)
    return p


def in_domain_cv(df, features, kind, groups_col="player_id"):
    """
    Honest in-domain reference: grouped CV, so the baseline the transfer number
    is compared against is itself out-of-sample.

    Fitting and scoring on the same rows would give ~0.99 and make the transfer
    gap look catastrophic for the wrong reason.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    X, y = df[features], df["y"].astype(int).to_numpy()
    groups = (df[groups_col].to_numpy() if groups_col in df.columns
              else np.arange(len(df)))
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=RANDOM_SEED)
    folds = []
    for tr, te in splitter.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        folds.append(evaluate(_fit(kind, features, X.iloc[tr], y[tr]),
                              X.iloc[te], y[te]))
    if not folds:
        return {}
    keys = [k for k in folds[0] if isinstance(folds[0][k], (int, float))]
    return {k: round(float(np.mean([f[k] for f in folds
                                    if f.get(k) is not None])), 4)
            for k in keys}


def cross_domain(syn, real, features, kind="xgb"):
    Xs, ys = syn[features], syn["y"].astype(int).to_numpy()
    Xr, yr = real[features], real["y"].astype(int).to_numpy()

    res = {}
    # in-domain references, cross-validated (grouped on player where possible)
    res["synth_in_domain_cv"] = in_domain_cv(syn, features, kind)
    res["real_in_domain_cv"]  = in_domain_cv(real, features, kind)

    # cross-domain: train on all of one domain, test on all of the other
    res["train_synth_test_real"] = evaluate(_fit(kind, features, Xs, ys), Xr, yr)
    res["train_real_test_synth"] = evaluate(_fit(kind, features, Xr, yr), Xs, ys)
    return res


# ─────────────────────────────────────────────────────────────
#  3. SHAP RANK AGREEMENT
# ─────────────────────────────────────────────────────────────

def shap_rank_agreement(syn, real, features, kind="xgb"):
    try:
        import shap
    except ImportError:
        print("  [skip] shap not installed")
        return None

    ranks = {}
    for name, d in (("synthetic", syn), ("real", real)):
        X, y = d[features], d["y"].astype(int).to_numpy()
        p = make_pipeline(kind, features)
        if kind == "xgb":
            p.named_steps["clf"].set_params(scale_pos_weight=scale_pos_weight(y))
        p.fit(X, y)
        Xt = pd.DataFrame(p.named_steps["pre"].transform(X), columns=features)
        sv = _train.tree_shap(p.named_steps["clf"], Xt)
        ranks[name] = pd.Series(np.abs(sv).mean(0), index=features)

    tab = pd.DataFrame(ranks)
    tab["rank_synthetic"] = tab["synthetic"].rank(ascending=False).astype(int)
    tab["rank_real"] = tab["real"].rank(ascending=False).astype(int)
    tab = tab.sort_values("rank_real")

    rho, p = spearmanr(tab["rank_synthetic"], tab["rank_real"])
    return tab, float(rho), float(p)


# ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", default=str(SYNTHETIC_CSV))
    ap.add_argument("--real", default=str(REAL_MODEL_CSV))
    ap.add_argument("--model", default="xgb", choices=["rf", "xgb"])
    ap.add_argument("--finetune", action="store_true",
                    help="pre-train BiLSTM on synthetic, fine-tune on real")
    ap.add_argument("--fast", action="store_true",
                    help="smaller models -- for iterating, not for final numbers")
    ap.add_argument("--sample", type=int, default=0,
                    help="subsample the real data to N rows (SHAP is slow on 14k)")
    args = ap.parse_args()

    if args.fast:
        _train.FAST = True
        print("[--fast] reduced model size; do not report these numbers.\n")

    syn, f_syn = load_synthetic(args.synthetic, TRANSFER_FEATURES)
    real, f_real = load_real(args.real, TRANSFER_FEATURES)
    if args.sample and len(real) > args.sample:
        real = real.sample(args.sample, random_state=RANDOM_SEED)
        print(f"[--sample] real data subsampled to {len(real):,} rows\n")
    shared = [f for f in TRANSFER_FEATURES if f in f_syn and f in f_real]

    print("=" * 72)
    print(f"synthetic: {len(syn):,} rows, {syn['y'].mean():.1%} positive")
    print(f"real     : {len(real):,} rows, {real['y'].mean():.1%} positive")
    print(f"shared features ({len(shared)}): {shared}")
    dropped = sorted(set(TRANSFER_FEATURES) - set(shared))
    if dropped:
        print(f"dropped (absent in one domain): {dropped}")
    print("=" * 72)

    if len(shared) < 3:
        raise SystemExit("Fewer than 3 shared features -- check TRANSFER_FEATURES "
                         "in config.py against your real dataset's columns.")

    out = {}

    print("\n[1] DISTRIBUTION MATCH")
    dm = distribution_match(syn, real, shared)
    print(dm.to_string(index=False))
    dm.to_csv(RESULTS_DIR / "transfer_distribution_match.csv", index=False)
    FIG_DIR.joinpath("transfer").mkdir(parents=True, exist_ok=True)
    plot_overlays(syn, real, shared,
                  str(FIG_DIR / "transfer" / "figT1_distributions.png"))
    print("\n  Interpret the effect sizes, not the p-values. |d| > 0.8 means "
          "the generator produced a materially different distribution for "
          "that variable -- name those variables in your Discussion.")

    print("\n[2] CROSS-DOMAIN TRANSFER")
    cd = cross_domain(syn, real, shared, args.model)
    out["cross_domain"] = cd
    for k, v in cd.items():
        if not v:
            continue
        print(f"  {k:26s} PR-AUC {v.get('pr_auc', float('nan')):.4f}  "
              f"(lift {v.get('pr_auc_lift', float('nan'))})  "
              f"bal-acc {v.get('balanced_accuracy', float('nan')):.4f}")

    try:
        ref  = cd["real_in_domain_cv"]["pr_auc_lift"]        # best achievable
        got  = cd["train_synth_test_real"]["pr_auc_lift"]    # synthetic-trained
        out["transfer_ratio"] = round(got / ref, 4) if ref else None
        print(f"\n  TRANSFER RATIO = {got:.3f} / {ref:.3f} = {got / ref:.3f}")
        print("  How much of the achievable real-data performance a "
              "synthetic-trained model recovers. Both terms are out-of-sample, "
              "so the comparison is fair. This single number is the cleanest "
              "expression of your contribution: near 1.0 means synthetic "
              "athlete data is a usable development proxy; near 0 means the "
              "generator does not reproduce the real fatigue signal.")
    except (KeyError, TypeError, ZeroDivisionError):
        pass

    print("\n[3] SHAP RANK AGREEMENT (RQ3)")
    sr = shap_rank_agreement(syn, real, shared, args.model)
    if sr:
        tab, rho, p = sr
        print(tab.round(4).to_string())
        tab.to_csv(RESULTS_DIR / "transfer_shap_ranks.csv")
        out["shap_spearman"] = {"rho": round(rho, 4), "p": p}
        print(f"\n  Spearman rho = {rho:.3f}  (p = {p:.3g})")
        if rho > 0.7:
            print("  Strong agreement: the generator's encoded sports-science "
                  "assumptions are broadly reproduced in real athletes.")
        elif rho > 0.3:
            print("  Moderate agreement: some assumptions transfer, others do "
                  "not. Identify which features moved most and discuss why.")
        else:
            print("  Weak agreement: the real data does not support the "
                  "feature structure your generator encoded. This is a strong "
                  "and honest finding -- lead your Discussion with it.")

    if args.finetune:
        print("\n[4] FINE-TUNING (optional)")
        print("  Not implemented as a one-liner on purpose: pre-train the "
              "BiLSTM from 08_train_real.py on the synthetic sequences, freeze "
              "nothing, then continue training on real sequences with a lower "
              "learning rate (1e-4) and compare against real-only training "
              "under identical grouped CV. Only attempt this if Day 23 ends "
              "early -- it is a bonus, not a requirement.")

    (RESULTS_DIR / "transfer_summary.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSummary -> {RESULTS_DIR / 'transfer_summary.json'}")


if __name__ == "__main__":
    main()
