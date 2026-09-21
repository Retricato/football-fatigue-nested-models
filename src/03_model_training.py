"""
=============================================================================
THESIS: AI-Driven Workload Analysis
=============================================================================
MODULE: 03_model_training.py
PURPOSE: Train Random Forest, XGBoost, and LSTM classifiers.
         Perform Stratified K-Fold + Leave-One-Player-Out (LOPO) validation.
         Compute classification metrics + ROC-AUC.
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import os
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.impute import SimpleImputer
import xgboost as xgb

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────

FEATURE_COLS = [
    # External load
    "total_distance_meters", "high_speed_running_distance", "sprint_distance",
    "acceleration_count", "deceleration_count", "player_load_score",
    "session_duration_minutes",
    # Internal load
    "resting_heart_rate", "average_training_hr", "max_training_hr",
    "HRV_rMSSD", "recovery_index",
    # Wellness
    "sleep_quality", "muscle_soreness", "mood_score", "RPE",
    # Derived
    "ACWR", "7_day_workload_average", "7_day_HRV_trend", "cumulative_fatigue_score",
    # Player context
    "age", "baseline_fitness_score",
]

TARGET_MULTICLASS  = "fatigue_state"          # Low / Moderate / High
TARGET_BINARY      = "next_day_fatigue_probability"   # → binarized at 0.5
TARGET_READINESS   = "recovery_readiness"     # Green / Amber / Red


def load_and_prepare(data_path: str):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["fatigue_state", "recovery_readiness"])

    # Encode targets
    le_fat = LabelEncoder()
    le_rec = LabelEncoder()
    df["fatigue_label"]   = le_fat.fit_transform(df["fatigue_state"])
    df["readiness_label"] = le_rec.fit_transform(df["recovery_readiness"])
    df["fatigue_binary"]  = (df["next_day_fatigue_probability"] > 0.5).astype(int)

    # Encode position
    df["position_encoded"] = LabelEncoder().fit_transform(df["playing_position"].fillna("Unknown"))
    features = FEATURE_COLS + ["position_encoded"]

    # Impute missing
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(df[features]), columns=features)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

    return df, X, X_scaled, le_fat, le_rec


# ─────────────────────────────────────────────
#  METRICS HELPER
# ─────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob=None, multiclass=True, label_names=None):
    avg = "macro" if multiclass else "binary"
    out = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, average=avg, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, average=avg, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, average=avg, zero_division=0), 4),
    }
    if y_prob is not None:
        try:
            if multiclass:
                out["roc_auc"] = round(
                    roc_auc_score(y_true, y_prob, multi_class="ovr",
                                  average="macro"), 4)
            else:
                out["roc_auc"] = round(roc_auc_score(y_true, y_prob[:, 1]), 4)
        except Exception:
            out["roc_auc"] = None
    return out


# ─────────────────────────────────────────────
#  STRATIFIED K-FOLD TRAINING
# ─────────────────────────────────────────────

def train_stratified_kfold(X, y, model_fn, n_splits=5, multiclass=True, label_names=None):
    """
    Stratified K-Fold cross-validation.
    Returns mean metrics and all fold predictions.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    all_preds = np.zeros(len(y), dtype=int)
    all_probs = np.zeros((len(y), len(np.unique(y))))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = model_fn()
        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        proba = model.predict_proba(X_val)

        all_preds[val_idx] = preds
        all_probs[val_idx] = proba

        m = compute_metrics(y_val, preds, proba, multiclass, label_names)
        fold_metrics.append(m)

    # Aggregate
    agg = {}
    for key in fold_metrics[0]:
        vals = [fm[key] for fm in fold_metrics if fm[key] is not None]
        agg[key] = {"mean": round(np.mean(vals), 4), "std": round(np.std(vals), 4)}

    return agg, all_preds, all_probs


# ─────────────────────────────────────────────
#  LEAVE-ONE-PLAYER-OUT VALIDATION
# ─────────────────────────────────────────────

def train_lopo(df, X, y, model_fn, multiclass=True):
    """
    Leave-One-Player-Out: train on all players except one, test on held-out player.
    Most realistic real-world evaluation for athlete monitoring models.
    """
    players = df["player_id"].values
    player_ids = np.unique(players)
    fold_metrics = []
    all_preds = np.zeros(len(y), dtype=int)

    for pid in player_ids:
        test_mask  = players == pid
        train_mask = ~test_mask

        X_tr = X[train_mask]
        X_val = X[test_mask]
        y_tr = y[train_mask]
        y_val = y[test_mask]

        if len(y_val) == 0 or len(np.unique(y_tr)) < 2:
            continue

        model = model_fn()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        all_preds[test_mask] = preds

        try:
            proba = model.predict_proba(X_val)
        except Exception:
            proba = None

        m = compute_metrics(y_val, preds, proba, multiclass)
        m["player"] = pid
        fold_metrics.append(m)

    return fold_metrics, all_preds


# ─────────────────────────────────────────────
#  LSTM MODEL
# ─────────────────────────────────────────────

def build_lstm(n_features: int, n_classes: int, seq_len: int = 7):
    """
    Bidirectional LSTM for time-series fatigue classification.
    Architecture: BiLSTM(64) → Dropout(0.3) → LSTM(32) → Dense(softmax)
    """
    import tensorflow as tf

    layers = tf.keras.layers
    Model = tf.keras.Model
    Input = tf.keras.Input

    inp = Input(shape=(seq_len, n_features))
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = Model(inp, out)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def prepare_sequences(df, X_scaled, y, seq_len: int = 7):
    """
    Create overlapping windows of length seq_len for LSTM input.
    Each sample is a sequence of consecutive days for one player.
    """
    Xs, ys = [], []
    for pid in df["player_id"].unique():
        mask = df["player_id"].values == pid
        X_p = X_scaled[mask].values
        y_p = y[mask].values
        for i in range(len(X_p) - seq_len):
            Xs.append(X_p[i:i + seq_len])
            ys.append(y_p[i + seq_len])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32)


def train_lstm(df, X_scaled, y, n_classes: int, seq_len: int = 7,
               epochs: int = 30, batch_size: int = 32):
    """
    Train LSTM with simple train/val split (80/20 random by sequence).
    """
    import tensorflow as tf
    tf.random.set_seed(42)

    Xs, ys = prepare_sequences(df, X_scaled, y, seq_len)
    n = len(Xs)
    idx = np.random.permutation(n)
    split = int(n * 0.8)
    tr_idx, val_idx = idx[:split], idx[split:]

    X_tr, y_tr = Xs[tr_idx], ys[tr_idx]
    X_val, y_val = Xs[val_idx], ys[val_idx]

    model = build_lstm(Xs.shape[2], n_classes, seq_len)

    cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                          monitor="val_accuracy")
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[cb],
        verbose=0,
    )

    y_pred_prob = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    metrics = compute_metrics(y_val, y_pred, y_pred_prob,
                              multiclass=(n_classes > 2))
    return model, history, metrics, y_val, y_pred


# ─────────────────────────────────────────────
#  SAVE METRICS
# ─────────────────────────────────────────────

def save_metrics_table(results: dict, out_dir: str):
    rows = []
    for model_name, target_results in results.items():
        for target_name, val_type_results in target_results.items():
            for val_type, metrics in val_type_results.items():
                if isinstance(metrics, dict) and "mean" not in str(metrics):
                    # LOPO list of per-player metrics
                    for m in metrics:
                        if isinstance(m, dict):
                            row = {
                                "model": model_name,
                                "target": target_name,
                                "validation": val_type,
                            }
                            row.update(m)
                            rows.append(row)
                else:
                    row = {"model": model_name, "target": target_name, "validation": val_type}
                    for k, v in metrics.items():
                        if isinstance(v, dict):
                            row[k] = f"{v['mean']:.4f} ± {v['std']:.4f}"
                        else:
                            row[k] = v
                    rows.append(row)
    summary_df = pd.DataFrame(rows)
    path = f"{out_dir}/model_metrics_summary.csv"
    summary_df.to_csv(path, index=False)
    print(f"  ✔ Metrics saved: {path}")
    return summary_df


# ─────────────────────────────────────────────
#  CONFUSION MATRIX PLOTS
# ─────────────────────────────────────────────

def plot_confusion_matrices(all_models_preds: dict, y_true_dict: dict,
                             le_dict: dict, out_dir: str):
    n_models = len(all_models_preds)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    fig.suptitle("Confusion Matrices — Fatigue State Classification (Stratified K-Fold)",
                 fontsize=12, fontweight="bold")

    for ax, (mname, preds) in zip(axes, all_models_preds.items()):
        y_true = y_true_dict[mname]
        le     = le_dict[mname]
        cm = confusion_matrix(y_true, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(mname)

    plt.tight_layout()
    path = f"{out_dir}/fig7_confusion_matrices.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved: {path}")


# ─────────────────────────────────────────────
#  LSTM TRAINING HISTORY PLOT
# ─────────────────────────────────────────────

def plot_lstm_history(history, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("LSTM Training History", fontsize=12, fontweight="bold")

    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="Train Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    path = f"{out_dir}/fig8_lstm_training_history.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved: {path}")


# ─────────────────────────────────────────────
#  MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────

def run_training(data_path: str = "../data/football_athlete_monitoring.csv",
                 results_dir: str = "../results",
                 viz_dir: str = "../visualizations"):

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    print("\n" + "="*60)
    print("  MODEL TRAINING PIPELINE")
    print("="*60)

    df, X, X_scaled, le_fat, le_rec = load_and_prepare(data_path)
    features = X.columns.tolist()

    y_fat  = df["fatigue_label"].reset_index(drop=True)
    y_bin  = df["fatigue_binary"].reset_index(drop=True)
    X_r    = X.reset_index(drop=True)
    X_sc_r = X_scaled.reset_index(drop=True)
    df_r   = df.reset_index(drop=True)

    results = {}
    cm_preds = {}
    cm_ytrue = {}
    cm_le    = {}

    # ── MODEL FACTORIES ────────────────────────────────────────────────────
    def rf_fn():
        return RandomForestClassifier(n_estimators=200, max_depth=10,
                                      min_samples_leaf=3, class_weight="balanced",
                                      random_state=42, n_jobs=-1)

    def xgb_fn():
        return xgb.XGBClassifier(n_estimators=200, max_depth=6,
                                  learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, use_label_encoder=False,
                                  eval_metric="mlogloss", random_state=42,
                                  verbosity=0)

    # ── A. RANDOM FOREST — FATIGUE STATE ───────────────────────────────────
    print("\n[1/3] Random Forest — Fatigue State (Stratified 5-Fold)...")
    rf_skf_metrics, rf_preds, rf_probs = train_stratified_kfold(
        X_r, y_fat, rf_fn, n_splits=5, multiclass=True,
        label_names=le_fat.classes_
    )
    print(f"  Accuracy: {rf_skf_metrics['accuracy']['mean']:.4f} "
          f"± {rf_skf_metrics['accuracy']['std']:.4f}")

    print("  → LOPO Validation...")
    rf_lopo, rf_lopo_preds = train_lopo(df_r, X_r, y_fat, rf_fn, multiclass=True)
    rf_lopo_mean = {
        k: round(np.mean([m[k] for m in rf_lopo if k in m and m[k] is not None]), 4)
        for k in ["accuracy", "precision", "recall", "f1"]
    }
    print(f"  LOPO Accuracy: {rf_lopo_mean['accuracy']:.4f}")

    results["RandomForest"] = {
        "fatigue_state": {
            "stratified_kfold": rf_skf_metrics,
            "lopo_mean": rf_lopo_mean,
        }
    }
    cm_preds["Random Forest"] = rf_preds
    cm_ytrue["Random Forest"] = y_fat.values
    cm_le["Random Forest"]    = le_fat

    # ── B. XGBOOST — FATIGUE STATE ─────────────────────────────────────────
    print("\n[2/3] XGBoost — Fatigue State (Stratified 5-Fold)...")
    xgb_skf_metrics, xgb_preds, xgb_probs = train_stratified_kfold(
        X_r, y_fat, xgb_fn, n_splits=5, multiclass=True
    )
    print(f"  Accuracy: {xgb_skf_metrics['accuracy']['mean']:.4f} "
          f"± {xgb_skf_metrics['accuracy']['std']:.4f}")

    print("  → LOPO Validation...")
    xgb_lopo, _ = train_lopo(df_r, X_r, y_fat, xgb_fn, multiclass=True)
    xgb_lopo_mean = {
        k: round(np.mean([m[k] for m in xgb_lopo if k in m and m[k] is not None]), 4)
        for k in ["accuracy", "precision", "recall", "f1"]
    }
    print(f"  LOPO Accuracy: {xgb_lopo_mean['accuracy']:.4f}")

    results["XGBoost"] = {
        "fatigue_state": {
            "stratified_kfold": xgb_skf_metrics,
            "lopo_mean": xgb_lopo_mean,
        }
    }
    cm_preds["XGBoost"] = xgb_preds
    cm_ytrue["XGBoost"] = y_fat.values
    cm_le["XGBoost"]    = le_fat

    # ── C. LSTM — FATIGUE STATE ────────────────────────────────────────────
    print("\n[3/3] LSTM (BiLSTM) — Fatigue State (80/20 temporal split)...")
    lstm_model, lstm_history, lstm_metrics, lstm_yval, lstm_ypred = train_lstm(
        df_r, X_sc_r, y_fat, n_classes=3, seq_len=7, epochs=40
    )
    print(f"  LSTM Accuracy: {lstm_metrics['accuracy']:.4f}  "
          f"F1: {lstm_metrics['f1']:.4f}")

    results["LSTM"] = {
        "fatigue_state": {
            "temporal_split_80_20": lstm_metrics
        }
    }

    # ── SAVE ARTIFACTS ─────────────────────────────────────────────────────
    plot_confusion_matrices(cm_preds, cm_ytrue, cm_le, viz_dir)
    plot_lstm_history(lstm_history, viz_dir)

    # Feature importances from RF and XGBoost
    rf_final = rf_fn()
    rf_final.fit(X_r, y_fat)
    xgb_final = xgb_fn()
    xgb_final.fit(X_r, y_fat)

    fi_rf  = pd.Series(rf_final.feature_importances_,  index=features).sort_values(ascending=False)
    fi_xgb = pd.Series(xgb_final.feature_importances_, index=features).sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Feature Importances — RF & XGBoost\n(Fatigue State Classification)",
                 fontsize=12, fontweight="bold")
    fi_rf.head(15).plot(kind="barh", ax=axes[0], color="#3498db", edgecolor="white")
    axes[0].invert_yaxis()
    axes[0].set_title("Random Forest")
    axes[0].set_xlabel("Importance")
    fi_xgb.head(15).plot(kind="barh", ax=axes[1], color="#e74c3c", edgecolor="white")
    axes[1].invert_yaxis()
    axes[1].set_title("XGBoost")
    axes[1].set_xlabel("Importance")
    plt.tight_layout()
    fi_path = f"{viz_dir}/fig9_feature_importances.png"
    plt.savefig(fi_path, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved: {fi_path}")

    # Save metrics JSON
    with open(f"{results_dir}/all_model_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  ✔ Metrics JSON saved: {results_dir}/all_model_metrics.json")

    # Print summary table
    print("\n" + "="*60)
    print("  RESULTS SUMMARY — Stratified K-Fold (Fatigue State)")
    print("="*60)
    print(f"{'Model':<16} {'Accuracy':>10} {'Precision':>12} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print("-"*60)
    for mname in ["RandomForest", "XGBoost"]:
        m = results[mname]["fatigue_state"]["stratified_kfold"]
        print(f"{mname:<16} "
              f"{m['accuracy']['mean']:>10.4f} "
              f"{m['precision']['mean']:>12.4f} "
              f"{m['recall']['mean']:>10.4f} "
              f"{m['f1']['mean']:>10.4f} "
              f"{str(m.get('roc_auc', {}).get('mean', 'N/A')):>10}")
    m = results["LSTM"]["fatigue_state"]["temporal_split_80_20"]
    print(f"{'LSTM':<16} "
          f"{m['accuracy']:>10.4f} "
          f"{m['precision']:>12.4f} "
          f"{m['recall']:>10.4f} "
          f"{m['f1']:>10.4f} "
          f"{str(m.get('roc_auc', 'N/A')):>10}")

    print("\n  LOPO Validation Summary:")
    print(f"{'Model':<16} {'Accuracy':>10} {'F1':>10}")
    print("-"*40)
    for mname in ["RandomForest", "XGBoost"]:
        m = results[mname]["fatigue_state"]["lopo_mean"]
        print(f"{mname:<16} {m['accuracy']:>10.4f} {m['f1']:>10.4f}")

    return results, rf_final, xgb_final, lstm_model, X_r, X_sc_r, y_fat, df_r, features


if __name__ == "__main__":
    run_training()
