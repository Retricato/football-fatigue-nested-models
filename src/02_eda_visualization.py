"""
=============================================================================
THESIS: AI-Driven Workload Analysis
=============================================================================
MODULE: 02_eda_visualization.py
PURPOSE: Exploratory Data Analysis + all thesis-quality visualizations
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

# ── Plot styling ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "legend.frameon":    False,
})

PALETTE = {
    "Low":      "#2ecc71",
    "Moderate": "#f39c12",
    "High":     "#e74c3c",
    "Green":    "#27ae60",
    "Amber":    "#e67e22",
    "Red":      "#c0392b",
}

SESSION_COLORS = {
    "REST":        "#bdc3c7",
    "ACTIVATION":  "#3498db",
    "LIGHT_LOAD":  "#2ecc71",
    "MEDIUM_LOAD": "#f39c12",
    "HIGH_LOAD":   "#e74c3c",
    "MATCH":       "#8e44ad",
}


def load_data(data_path: str = "../data/football_athlete_monitoring.csv") -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 1: CORRELATION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_matrix(df: pd.DataFrame, out_dir: str):
    features = [
        "total_distance_meters", "high_speed_running_distance", "sprint_distance",
        "acceleration_count", "player_load_score", "session_duration_minutes",
        "resting_heart_rate", "HRV_rMSSD", "recovery_index",
        "sleep_quality", "muscle_soreness", "mood_score", "RPE",
        "ACWR", "7_day_workload_average", "7_day_HRV_trend",
        "cumulative_fatigue_score", "next_day_fatigue_probability",
    ]
    sub = df[features].dropna()
    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 6.5},
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})

    ax.set_title("Feature Correlation Matrix — Football Athlete Monitoring Dataset\n"
                 "(Lower Triangle | Pearson r)", fontsize=13, pad=15, fontweight="bold")
    plt.tight_layout()
    path = f"{out_dir}/fig1_correlation_matrix.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 2: WORKLOAD TRENDS — TEAM OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

def plot_workload_trends(df: pd.DataFrame, out_dir: str):
    team_daily = df.groupby("day_number").agg({
        "total_distance_meters":      "mean",
        "high_speed_running_distance": "mean",
        "sprint_distance":            "mean",
        "ACWR":                       "mean",
        "7_day_workload_average":     "mean",
        "session_type":               "first",
    }).reset_index()

    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    fig.suptitle("Team Workload Trends Over Monitoring Period",
                 fontsize=13, fontweight="bold", y=1.01)

    x = team_daily["day_number"]

    # Panel 1: Distance metrics
    ax = axes[0]
    ax.fill_between(x, team_daily["total_distance_meters"] / 1000,
                    alpha=0.3, color="#3498db", label="Total Distance (km)")
    ax.plot(x, team_daily["total_distance_meters"] / 1000,
            color="#3498db", lw=1.8)
    ax.fill_between(x, team_daily["high_speed_running_distance"],
                    alpha=0.4, color="#e74c3c", label="HSR (m)")
    ax.plot(x, team_daily["high_speed_running_distance"],
            color="#e74c3c", lw=1.4, ls="--")
    ax.set_ylabel("Distance")
    ax.legend(loc="upper right", fontsize=9)

    # Match day markers
    match_days = team_daily[team_daily["session_type"] == "MATCH"]["day_number"]
    for md in match_days:
        ax.axvline(md, color="#8e44ad", lw=1.2, ls=":", alpha=0.7)

    # Panel 2: Sprint distance
    ax = axes[1]
    ax.bar(x, team_daily["sprint_distance"], color="#f39c12", alpha=0.75,
           label="Sprint Distance (m)", width=0.85)
    for md in match_days:
        ax.axvline(md, color="#8e44ad", lw=1.2, ls=":", alpha=0.7)
    ax.set_ylabel("Sprint Distance (m)")
    ax.legend(fontsize=9)

    # Panel 3: ACWR with risk zones
    ax = axes[2]
    ax.axhspan(0.8, 1.3, alpha=0.12, color="#2ecc71", label="Optimal zone (0.8–1.3)")
    ax.axhspan(1.3, 1.8, alpha=0.12, color="#e74c3c", label="Danger zone (>1.3)")
    ax.axhspan(0.0, 0.8, alpha=0.08, color="#f39c12", label="Under-loaded (<0.8)")
    ax.plot(x, team_daily["ACWR"], color="#2c3e50", lw=2, label="Team ACWR")
    ax.plot(x, team_daily["7_day_workload_average"],
            color="#3498db", lw=1.5, ls="--", label="7-day Workload Avg")
    ax.axhline(1.0, color="grey", lw=1, ls="--", alpha=0.5)
    for md in match_days:
        ax.axvline(md, color="#8e44ad", lw=1.2, ls=":", alpha=0.7)
    ax.set_ylabel("ACWR / Load AU")
    ax.set_xlabel("Training Day")
    ax.legend(fontsize=8, ncol=3)

    # Add session type color band at bottom
    for _, row in team_daily.iterrows():
        axes[0].axvspan(row["day_number"] - 0.5, row["day_number"] + 0.5,
                        alpha=0.07,
                        color=SESSION_COLORS.get(row["session_type"], "white"))

    match_legend = Line2D([0], [0], color="#8e44ad", lw=1.5, ls=":", label="Match Day")
    axes[0].legend(handles=list(axes[0].get_legend().legend_handles) + [match_legend],
                   fontsize=9)

    plt.tight_layout()
    path = f"{out_dir}/fig2_workload_trends.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 3: HRV vs WORKLOAD TRENDS (per player subset)
# ─────────────────────────────────────────────────────────────────────────────

def plot_hrv_workload(df: pd.DataFrame, out_dir: str, n_players: int = 4):
    players = df["player_id"].unique()[:n_players]
    fig, axes = plt.subplots(n_players, 1, figsize=(16, 4 * n_players), sharex=False)
    if n_players == 1:
        axes = [axes]

    fig.suptitle("HRV (rMSSD) vs Workload — Individual Player Analysis\n"
                 "(Inverse relationship reflects autonomic stress response)",
                 fontsize=13, fontweight="bold")

    for ax, pid in zip(axes, players):
        sub = df[df["player_id"] == pid].dropna(subset=["HRV_rMSSD"]).reset_index()
        pos = sub["playing_position"].iloc[0]

        ax2 = ax.twinx()

        # Load bars
        ax2.bar(sub["day_number"], sub["session_load_au"] if "session_load_au" in sub.columns
                else sub["player_load_score"].fillna(0),
                alpha=0.25, color="#3498db", width=0.9, label="Session Load (AU)")

        # HRV line
        hrv_smooth = gaussian_filter_safe(sub["HRV_rMSSD"].values, sigma=1.5)
        ax.plot(sub["day_number"], hrv_smooth,
                color="#e74c3c", lw=2, label="HRV rMSSD (ms)")
        ax.scatter(sub["day_number"], sub["HRV_rMSSD"],
                   color="#e74c3c", s=15, alpha=0.5, zorder=3)

        # Match markers
        match_days = sub[sub["session_type"] == "MATCH"]["day_number"]
        for md in match_days:
            ax.axvline(md, color="#8e44ad", lw=1.2, ls=":", alpha=0.6)

        ax.set_ylabel("HRV rMSSD (ms)", color="#e74c3c")
        ax2.set_ylabel("Session Load (AU)", color="#3498db")
        ax.set_title(f"{pid} — {pos}", fontsize=10, loc="left")
        ax.tick_params(axis="y", colors="#e74c3c")
        ax2.tick_params(axis="y", colors="#3498db")

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Training Day")
    plt.tight_layout()
    path = f"{out_dir}/fig3_hrv_vs_workload.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved: {path}")


def gaussian_filter_safe(arr, sigma=1.5):
    from scipy.ndimage import gaussian_filter1d
    valid = ~np.isnan(arr)
    result = arr.copy()
    if valid.sum() > 3:
        result[valid] = gaussian_filter1d(arr[valid], sigma=sigma)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 4: FATIGUE TIMELINE — ALL PLAYERS
# ─────────────────────────────────────────────────────────────────────────────

def plot_fatigue_timeline(df: pd.DataFrame, out_dir: str):
    players = sorted(df["player_id"].unique())
    n = len(players)
    fig, axes = plt.subplots(n, 1, figsize=(18, 2.8 * n), sharex=False)
    if n == 1:
        axes = [axes]

    fig.suptitle("Player Fatigue State Timeline Over Monitoring Period",
                 fontsize=13, fontweight="bold")

    state_to_num = {"Low": 0, "Moderate": 1, "High": 2}
    cmap_colors  = ["#2ecc71", "#f39c12", "#e74c3c"]

    for ax, pid in zip(axes, players):
        sub = df[df["player_id"] == pid].reset_index(drop=True)
        pos = sub["playing_position"].iloc[0]

        fatigue_num = sub["fatigue_state"].map(state_to_num).fillna(0).astype(int)

        for i in range(len(sub)):
            ax.bar(sub["day_number"].iloc[i], 1.0,
                   color=cmap_colors[fatigue_num.iloc[i]],
                   alpha=0.75, width=1.0, align="center")

        # Recovery readiness overlay (line)
        readiness_map = {"Green": 0.85, "Amber": 0.50, "Red": 0.15}
        readiness_y = sub["recovery_readiness"].map(readiness_map).fillna(0.5)
        ax.plot(sub["day_number"], readiness_y, color="#2c3e50",
                lw=1.5, ls="--", alpha=0.7, label="Recovery Readiness")

        # Match day markers
        for md in sub[sub["session_type"] == "MATCH"]["day_number"]:
            ax.axvline(md, color="#8e44ad", lw=1.5, ls="--", alpha=0.8)

        ax.set_xlim(0, sub["day_number"].max() + 1)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([])
        ax.set_title(f"{pid} ({pos})", fontsize=9, loc="left", pad=3)
        ax.set_xlabel("Day")

        # Fatigue probability overlay
        ax2 = ax.twinx()
        ax2.plot(sub["day_number"], sub["next_day_fatigue_probability"],
                 color="#8e44ad", lw=1.2, alpha=0.6, label="P(fatigue)")
        ax2.set_ylim(0, 1)
        ax2.set_yticks([0, 0.5, 1])
        ax2.tick_params(labelsize=7)
        ax2.set_ylabel("P(fatigue)", fontsize=7, color="#8e44ad")

    # Legend
    patches = [
        mpatches.Patch(color="#2ecc71", label="Low Fatigue"),
        mpatches.Patch(color="#f39c12", label="Moderate Fatigue"),
        mpatches.Patch(color="#e74c3c", label="High Fatigue"),
        Line2D([0], [0], color="#8e44ad", lw=1.5, ls="--", label="Match Day"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    path = f"{out_dir}/fig4_fatigue_timeline.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 5: FATIGUE RISK DASHBOARD (summary)
# ─────────────────────────────────────────────────────────────────────────────

def plot_fatigue_risk_dashboard(df: pd.DataFrame, out_dir: str):
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Fatigue Risk Dashboard — Team Summary",
                 fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel A: Fatigue state distribution by player ──────────────────────
    ax_a = fig.add_subplot(gs[0, :2])
    fat_counts = (df.groupby(["player_id", "fatigue_state"])
                    .size()
                    .unstack(fill_value=0)
                    .reindex(columns=["Low", "Moderate", "High"], fill_value=0))
    fat_counts.plot(kind="bar", stacked=True, ax=ax_a,
                    color=["#2ecc71", "#f39c12", "#e74c3c"],
                    edgecolor="white", width=0.7)
    ax_a.set_title("A — Fatigue State Distribution per Player")
    ax_a.set_xlabel("")
    ax_a.set_ylabel("Days")
    ax_a.tick_params(axis="x", rotation=0)
    ax_a.legend(title="Fatigue State", fontsize=8)

    # ── Panel B: Recovery readiness pie ────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 2])
    rc = df["recovery_readiness"].value_counts()
    colors_pie = [PALETTE.get(k, "#aaa") for k in rc.index]
    ax_b.pie(rc.values, labels=rc.index, colors=colors_pie,
             autopct="%1.1f%%", startangle=90,
             textprops={"fontsize": 9})
    ax_b.set_title("B — Recovery Readiness\n(All player-days)")

    # ── Panel C: ACWR box per week ─────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, :2])
    acwr_data = [df[df["week_number"] == w]["ACWR"].dropna().values
                 for w in sorted(df["week_number"].unique())]
    bp = ax_c.boxplot(acwr_data, patch_artist=True,
                      medianprops={"color": "black", "lw": 2})
    colors_box = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(acwr_data)))
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax_c.axhline(1.3, color="#e74c3c", ls="--", lw=1.5, alpha=0.8, label="Risk threshold (1.3)")
    ax_c.axhline(0.8, color="#f39c12", ls="--", lw=1.5, alpha=0.8, label="Under-load (0.8)")
    ax_c.set_xticklabels([f"Wk {w}" for w in sorted(df["week_number"].unique())])
    ax_c.set_ylabel("ACWR")
    ax_c.set_title("C — Weekly ACWR Distribution")
    ax_c.legend(fontsize=8)

    # ── Panel D: HRV by fatigue state ─────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 2])
    for state, color in [("Low", "#2ecc71"), ("Moderate", "#f39c12"), ("High", "#e74c3c")]:
        vals = df[df["fatigue_state"] == state]["HRV_rMSSD"].dropna()
        ax_d.hist(vals, bins=20, alpha=0.55, color=color, label=state, edgecolor="white")
    ax_d.set_xlabel("HRV rMSSD (ms)")
    ax_d.set_ylabel("Frequency")
    ax_d.set_title("D — HRV Distribution\nby Fatigue State")
    ax_d.legend(fontsize=8)

    # ── Panel E: Fatigue probability heatmap ──────────────────────────────
    ax_e = fig.add_subplot(gs[2, :])
    heat = df.pivot_table(index="player_id", columns="week_number",
                           values="next_day_fatigue_probability",
                           aggfunc="mean")
    sns.heatmap(heat, cmap="RdYlGn_r", vmin=0, vmax=1, ax=ax_e,
                annot=True, fmt=".2f", linewidths=0.5,
                cbar_kws={"label": "Mean P(fatigue next day)"})
    ax_e.set_title("E — Mean Next-Day Fatigue Probability Heatmap (Player × Week)")
    ax_e.set_xlabel("Week Number")
    ax_e.set_ylabel("Player")

    path = f"{out_dir}/fig5_fatigue_risk_dashboard.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE 6: WELLNESS METRICS OVER TIME
# ─────────────────────────────────────────────────────────────────────────────

def plot_wellness_trends(df: pd.DataFrame, out_dir: str):
    team = df.groupby("day_number").agg({
        "sleep_quality":   ["mean", "std"],
        "muscle_soreness": ["mean", "std"],
        "mood_score":      ["mean", "std"],
        "RPE":             "mean",
        "session_type":    "first",
    }).reset_index()

    team.columns = ["day_number",
                    "sleep_mean", "sleep_std",
                    "soreness_mean", "soreness_std",
                    "mood_mean", "mood_std",
                    "rpe_mean", "session_type"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=False)
    fig.suptitle("Team Wellness & Subjective Load Trends",
                 fontsize=13, fontweight="bold")

    metrics = [
        ("sleep_mean",    "sleep_std",    "Sleep Quality (1–10)", "#3498db", axes[0, 0]),
        ("soreness_mean", "soreness_std", "Muscle Soreness (1–10)", "#e74c3c", axes[0, 1]),
        ("mood_mean",     "mood_std",     "Mood Score (1–10)", "#2ecc71", axes[1, 0]),
    ]

    for col_m, col_s, title, color, ax in metrics:
        x = team["day_number"]
        ax.fill_between(x,
                        team[col_m] - team[col_s],
                        team[col_m] + team[col_s],
                        alpha=0.2, color=color)
        ax.plot(x, team[col_m], color=color, lw=2)
        match_days = team[team["session_type"] == "MATCH"]["day_number"]
        for md in match_days:
            ax.axvline(md, color="#8e44ad", lw=1.1, ls=":", alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("Training Day")
        ax.set_ylabel("Score")
        ax.set_ylim(1, 10)

    # RPE bar chart
    ax = axes[1, 1]
    rpe_vals = team.dropna(subset=["rpe_mean"])
    ax.bar(rpe_vals["day_number"], rpe_vals["rpe_mean"],
           color="#9b59b6", alpha=0.75, width=0.85)
    for md in team[team["session_type"] == "MATCH"]["day_number"]:
        ax.axvline(md, color="#8e44ad", lw=1.2, ls="--", alpha=0.7)
    ax.set_title("Team Mean RPE (1–10)")
    ax.set_xlabel("Training Day")
    ax.set_ylabel("RPE")
    ax.set_ylim(0, 10)

    plt.tight_layout()
    path = f"{out_dir}/fig6_wellness_trends.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✔ Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_eda(data_path: str = "../data/football_athlete_monitoring.csv",
            out_dir:  str = "../visualizations"):
    os.makedirs(out_dir, exist_ok=True)
    df = load_data(data_path)

    print("\n" + "="*60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("="*60)
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"  Missing values: {df.isnull().sum().sum()} "
          f"({df.isnull().mean().mean()*100:.1f}% overall)")
    for col in ("fatigue_state", "recovery_readiness", "y"):
        if col in df.columns:
            print(f"\n  {col} counts:\n{df[col].value_counts()}")

    print("\nGenerating figures...")
    # Each plot is guarded: the real dataset has no HRV, no recovery_readiness
    # and no playing_position, so those figures are skipped rather than crashing.
    for name, fn in (
        ("correlation_matrix",   lambda: plot_correlation_matrix(df, out_dir)),
        ("workload_trends",      lambda: plot_workload_trends(df, out_dir)),
        ("hrv_vs_workload",      lambda: plot_hrv_workload(df, out_dir, n_players=4)),
        ("fatigue_timeline",     lambda: plot_fatigue_timeline(df, out_dir)),
        ("fatigue_risk_dashboard", lambda: plot_fatigue_risk_dashboard(df, out_dir)),
        ("wellness_trends",      lambda: plot_wellness_trends(df, out_dir)),
    ):
        try:
            fn()
        except (KeyError, ValueError) as exc:
            print(f"  [skip] {name}: {type(exc).__name__}: {exc}")
            print(f"         (expected on real data -- variable not available)")

    print(f"\nEDA figures saved to {out_dir}")
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="EDA figures. Works on both the synthetic and the real dataset.")
    ap.add_argument("--data", default="football_athlete_monitoring.csv")
    ap.add_argument("--out", default="figures/synthetic")
    a = ap.parse_args()
    run_eda(a.data, a.out)
