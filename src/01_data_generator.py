"""
=============================================================================
THESIS: AI-Driven Workload Analysis: Predicting Fatigue and Recovery Dynamics
        in Football via Machine Learning
=============================================================================
MODULE: 01_data_generator.py
PURPOSE: Generate realistic synthetic longitudinal football athlete data
AUTHOR: Master's Thesis Project — Sports Data Science
=============================================================================

Domain references:
- Gabbett (2016): Acute:chronic workload ratio framework
- Buchheit & Laursen (2013): HRV monitoring in athletes
- Halson (2014): Monitoring training load to understand fatigue
- FIFA EPTS (2015): Electronic Performance & Tracking Systems standards
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import warnings
import os

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────
#  PLAYER ARCHETYPES
# ─────────────────────────────────────────────

POSITIONS = ["Defender", "Midfielder", "Forward", "Goalkeeper"]

# Position-specific GPS load multipliers (based on published norms)
POSITION_LOAD_PROFILE = {
    "Midfielder":  {"distance": 1.10, "hsr": 1.05, "sprint": 0.95, "accel": 1.10},
    "Forward":     {"distance": 0.95, "hsr": 1.10, "sprint": 1.20, "accel": 1.10},
    "Defender":    {"distance": 1.00, "hsr": 1.00, "sprint": 1.00, "accel": 1.00},
    "Goalkeeper":  {"distance": 0.40, "hsr": 0.25, "sprint": 0.20, "accel": 0.70},
}

def create_player_profiles(n_players: int = 9) -> pd.DataFrame:
    """
    Create inter-individually variable player profiles.
    Captures real-world diversity in age, fitness, fatigue sensitivity,
    and recovery capacity.
    """
    np.random.seed(42)
    positions = (POSITIONS * 3)[:n_players]
    np.random.shuffle(positions)

    players = []
    for i in range(n_players):
        pos = positions[i]
        age = int(np.random.normal(24, 3.5))
        age = np.clip(age, 18, 32)

        # Baseline fitness: younger midfielders tend higher, GKs lower
        base_fitness = np.random.normal(72, 8)
        if pos == "Midfielder":
            base_fitness += 5
        elif pos == "Goalkeeper":
            base_fitness -= 8
        base_fitness = np.clip(base_fitness, 50, 95)

        # Recovery rate: inter-player variability (0.6 = slow, 1.4 = fast)
        recovery_rate = np.random.normal(1.0, 0.2)
        recovery_rate = np.clip(recovery_rate, 0.6, 1.4)

        # Fatigue sensitivity (amplifies fatigue response)
        fatigue_sensitivity = np.random.normal(1.0, 0.25)
        fatigue_sensitivity = np.clip(fatigue_sensitivity, 0.5, 1.8)

        # HRV baseline (rMSSD ms) — influenced by age and fitness
        hrv_baseline = 60 - (age - 18) * 0.8 + (base_fitness - 70) * 0.5
        hrv_baseline += np.random.normal(0, 5)
        hrv_baseline = np.clip(hrv_baseline, 30, 95)

        players.append({
            "player_id": f"P{i+1:02d}",
            "age": age,
            "playing_position": pos,
            "baseline_fitness_score": round(base_fitness, 1),
            "recovery_rate": round(recovery_rate, 3),       # hidden feature
            "fatigue_sensitivity": round(fatigue_sensitivity, 3),  # hidden
            "hrv_baseline": round(hrv_baseline, 1),         # hidden
        })

    return pd.DataFrame(players)


# ─────────────────────────────────────────────
#  SESSION TYPE SCHEDULE
# ─────────────────────────────────────────────

def generate_weekly_schedule(n_weeks: int) -> list:
    """
    Return a list of session types for each day.
    Typical professional football microcycle (Monday = day after match):
      Mon: Recovery/REST, Tue: Activation, Wed: MD-3 (high load),
      Thu: MD-2 (medium), Fri: MD-1 (light), Sat: MATCH, Sun: REST
    """
    microcycle = [
        "REST",         # Day 1 (post-match recovery)
        "ACTIVATION",   # Day 2
        "HIGH_LOAD",    # Day 3 (peak volume)
        "MEDIUM_LOAD",  # Day 4
        "LIGHT_LOAD",   # Day 5 (pre-match)
        "MATCH",        # Day 6 (match day)
        "REST",         # Day 7
    ]
    schedule = []
    for _ in range(n_weeks):
        schedule.extend(microcycle)
    return schedule


SESSION_LOAD_PARAMS = {
    "REST":        {"dist_mean": 0,     "dist_sd": 0,    "duration": 0,  "intensity": 0.0},
    "ACTIVATION":  {"dist_mean": 4500,  "dist_sd": 400,  "duration": 45, "intensity": 0.4},
    "LIGHT_LOAD":  {"dist_mean": 5500,  "dist_sd": 500,  "duration": 55, "intensity": 0.5},
    "MEDIUM_LOAD": {"dist_mean": 8000,  "dist_sd": 700,  "duration": 75, "intensity": 0.7},
    "HIGH_LOAD":   {"dist_mean": 10500, "dist_sd": 900,  "duration": 90, "intensity": 0.85},
    "MATCH":       {"dist_mean": 11000, "dist_sd": 800,  "duration": 95, "intensity": 1.0},
}


# ─────────────────────────────────────────────
#  EXTERNAL LOAD GENERATOR
# ─────────────────────────────────────────────

def generate_external_load(session_type: str, position: str,
                            fatigue_level: float, day: int) -> dict:
    """
    Generate GPS external load metrics per session.
    Accounts for: session type, position profile, fatigue suppression.

    Published norms (Malone et al., 2017; Rampinini et al., 2007):
    - Match total distance: 10–12 km (outfield); 5–7 km (GK)
    - HSR (>19.8 km/h): 800–1200 m (outfield)
    - Sprint (>25.2 km/h): 200–400 m (outfield)
    """
    params = SESSION_LOAD_PARAMS[session_type]
    pos_profile = POSITION_LOAD_PROFILE[position]

    if session_type == "REST":
        return {
            "total_distance_meters": 0,
            "high_speed_running_distance": 0,
            "sprint_distance": 0,
            "acceleration_count": 0,
            "deceleration_count": 0,
            "player_load_score": 0,
            "session_duration_minutes": 0,
        }

    # Fatigue suppresses output (reduced voluntary effort)
    fatigue_suppression = 1.0 - (fatigue_level * 0.15)

    # Base distance
    base_dist = np.random.normal(params["dist_mean"], params["dist_sd"])
    base_dist *= pos_profile["distance"] * fatigue_suppression
    base_dist = max(0, base_dist)

    # HSR: ~8-12% of total distance (outfield), less for GK
    hsr_fraction = np.random.normal(0.09, 0.015) * pos_profile["hsr"]
    hsr = base_dist * hsr_fraction

    # Sprint: ~2-4% of total distance
    sprint_fraction = np.random.normal(0.028, 0.008) * pos_profile["sprint"]
    sprint_dist = base_dist * sprint_fraction

    # Accelerations/decelerations (>3 m/s²)
    accel_base = params["intensity"] * np.random.normal(38, 8) * pos_profile["accel"]
    accel_count = max(0, int(accel_base * fatigue_suppression))
    decel_count = max(0, int(accel_count * np.random.normal(0.95, 0.05)))

    # Player Load Score (arbitrary units, composite)
    player_load = (base_dist / 1000) * params["intensity"] * np.random.normal(1.8, 0.15)
    player_load *= fatigue_suppression

    duration = params["duration"] + np.random.normal(0, 5)
    duration = max(0, duration)

    return {
        "total_distance_meters": round(base_dist, 1),
        "high_speed_running_distance": round(max(0, hsr), 1),
        "sprint_distance": round(max(0, sprint_dist), 1),
        "acceleration_count": accel_count,
        "deceleration_count": decel_count,
        "player_load_score": round(max(0, player_load), 2),
        "session_duration_minutes": round(max(0, duration), 1),
    }


# ─────────────────────────────────────────────
#  INTERNAL LOAD GENERATOR
# ─────────────────────────────────────────────

def generate_internal_load(session_type: str, fatigue_level: float,
                            hrv_baseline: float, recovery_rate: float) -> dict:
    """
    Generate physiological internal load metrics.
    HRV (rMSSD) decreases with fatigue accumulation.
    HR responses increase under fatigue (autonomic dysregulation).
    """
    if session_type == "REST":
        # On rest days: resting HR and HRV recovery
        rhr = np.random.normal(55, 5) + fatigue_level * 3
        hrv = hrv_baseline * (1 - fatigue_level * 0.15) * np.random.normal(1, 0.05)
        return {
            "resting_heart_rate": round(np.clip(rhr, 40, 85), 0),
            "average_training_hr": np.nan,
            "max_training_hr": np.nan,
            "HRV_rMSSD": round(np.clip(hrv, 20, 100), 1),
            "recovery_index": round(np.clip(hrv / hrv_baseline, 0.5, 1.2), 3),
        }

    params = SESSION_LOAD_PARAMS[session_type]
    intensity = params["intensity"]

    # Resting HR elevates with fatigue
    rhr = np.random.normal(55, 4) + fatigue_level * 4
    rhr = np.clip(rhr, 40, 85)

    # Training HR scales with intensity and fatigue
    avg_hr = 100 + intensity * 75 + fatigue_level * 8 + np.random.normal(0, 5)
    avg_hr = np.clip(avg_hr, 95, 185)

    max_hr = avg_hr + np.random.normal(22, 4) + intensity * 10
    max_hr = np.clip(max_hr, avg_hr + 5, 205)

    # HRV decreases under fatigue (Plews et al., 2013)
    hrv_suppression = 1 - (fatigue_level * 0.20) - (intensity * 0.10)
    hrv = hrv_baseline * hrv_suppression * np.random.normal(1, 0.06)
    hrv = np.clip(hrv, 15, 100)

    # Recovery index: ratio of current HRV to baseline
    recovery_index = hrv / hrv_baseline
    recovery_index = np.clip(recovery_index, 0.4, 1.3)

    return {
        "resting_heart_rate": round(rhr, 0),
        "average_training_hr": round(avg_hr, 0),
        "max_training_hr": round(max_hr, 0),
        "HRV_rMSSD": round(hrv, 1),
        "recovery_index": round(recovery_index, 3),
    }


# ─────────────────────────────────────────────
#  SUBJECTIVE WELLNESS GENERATOR
# ─────────────────────────────────────────────

def generate_wellness(session_type: str, fatigue_level: float,
                       yesterday_session: str, recovery_rate: float) -> dict:
    """
    Generate subjective wellness scores (Hooper Index style).
    Scores are inversely correlated with fatigue.
    RPE reflects session intensity and accumulated fatigue.
    """
    # Sleep quality: lower after high load / match
    if yesterday_session in ["HIGH_LOAD", "MATCH"]:
        sleep_base = np.random.normal(5.5, 1.2)
    elif yesterday_session == "REST":
        sleep_base = np.random.normal(7.5, 0.8)
    else:
        sleep_base = np.random.normal(6.8, 1.0)

    sleep_quality = sleep_base - (fatigue_level * 1.5) + np.random.normal(0, 0.5)
    sleep_quality = np.clip(sleep_quality, 1, 10)

    # Muscle soreness: higher after high load, match
    if yesterday_session in ["HIGH_LOAD", "MATCH"]:
        soreness_base = np.random.normal(6.0, 1.2)
    elif yesterday_session == "REST":
        soreness_base = np.random.normal(3.0, 1.0)
    else:
        soreness_base = np.random.normal(4.5, 1.0)

    soreness = soreness_base + (fatigue_level * 2.0) + np.random.normal(0, 0.5)
    soreness = np.clip(soreness, 1, 10)

    # Mood: inversely related to fatigue
    mood = np.random.normal(7.0, 1.0) - (fatigue_level * 2.5) + np.random.normal(0, 0.3)
    mood = np.clip(mood, 1, 10)

    # RPE: only on training/match days
    if session_type == "REST":
        rpe = np.nan
    else:
        intensity = SESSION_LOAD_PARAMS[session_type]["intensity"]
        rpe = intensity * 9 + fatigue_level * 1.5 + np.random.normal(0, 0.6)
        rpe = np.clip(rpe, 1, 10)

    return {
        "sleep_quality": round(sleep_quality, 1),
        "muscle_soreness": round(soreness, 1),
        "mood_score": round(mood, 1),
        "RPE": round(rpe, 1) if not np.isnan(rpe) else np.nan,
    }


# ─────────────────────────────────────────────
#  FATIGUE STATE ENGINE
# ─────────────────────────────────────────────

def update_fatigue(current_fatigue: float, session_type: str,
                   hrv_rmssd: float, hrv_baseline: float,
                   sleep_quality: float, soreness: float,
                   recovery_rate: float, fatigue_sensitivity: float) -> float:
    """
    Domain-informed fatigue state update rule.

    Implements a simplified impulse-response model:
      fatigue(t+1) = fatigue(t) * decay + load_impulse(t)

    Rules grounded in:
    - Banister (1991) fitness-fatigue model
    - Gabbett (2016) ACWR
    - Plews (2013) HRV monitoring
    """
    decay = 0.80 * recovery_rate   # faster recovery → steeper decay

    # Session load impulse
    load_impulse = {
        "REST":        -0.08,
        "ACTIVATION":   0.10,
        "LIGHT_LOAD":   0.18,
        "MEDIUM_LOAD":  0.30,
        "HIGH_LOAD":    0.45,
        "MATCH":        0.55,
    }[session_type]

    # HRV penalty: low HRV → higher fatigue
    hrv_ratio = hrv_rmssd / hrv_baseline if hrv_baseline > 0 else 1.0
    hrv_penalty = max(0, (1 - hrv_ratio) * 0.15)

    # Wellness adjustment
    wellness_factor = ((sleep_quality / 10) * 0.5 + (1 - soreness / 10) * 0.5)
    wellness_adjustment = (0.5 - wellness_factor) * 0.10

    # Apply sensitivity multiplier
    raw_delta = (load_impulse + hrv_penalty + wellness_adjustment) * fatigue_sensitivity

    new_fatigue = current_fatigue * decay + raw_delta
    return float(np.clip(new_fatigue, 0.0, 1.0))


# ─────────────────────────────────────────────
#  TARGET LABEL GENERATION
# ─────────────────────────────────────────────

def assign_fatigue_state(fatigue: float) -> str:
    if fatigue < 0.33:
        return "Low"
    elif fatigue < 0.66:
        return "Moderate"
    else:
        return "High"


def assign_recovery_readiness(fatigue: float, hrv_ratio: float,
                               sleep: float, soreness: float) -> str:
    """
    Multi-factor readiness score following Haddad et al. (2017).
    """
    score = (1 - fatigue) * 0.4 + hrv_ratio * 0.3 + (sleep / 10) * 0.2 + (1 - soreness / 10) * 0.1
    if score > 0.70:
        return "Green"
    elif score > 0.45:
        return "Amber"
    else:
        return "Red"


def next_day_fatigue_prob(fatigue: float, hrv_ratio: float,
                          soreness: float, rpe: float,
                          consecutive_load_days: int) -> float:
    """
    Probabilistic output: estimated probability of high fatigue next day.
    Uses logistic-style transform with domain rules.
    """
    rpe_norm = rpe / 10 if not np.isnan(rpe) else 0.5

    # Rule: consecutive high-load days amplify probability
    consecutive_bonus = min(consecutive_load_days * 0.08, 0.25)

    # Rule: HRV drop + high RPE = elevated risk
    hrv_penalty = max(0, (1 - hrv_ratio) * 0.3)

    raw = (fatigue * 0.45
           + (1 - hrv_ratio) * 0.20
           + (soreness / 10) * 0.15
           + rpe_norm * 0.10
           + hrv_penalty
           + consecutive_bonus)

    # Sigmoid squash
    prob = 1 / (1 + np.exp(-8 * (raw - 0.5)))
    return round(float(np.clip(prob, 0.01, 0.99)), 4)


# ─────────────────────────────────────────────
#  DERIVED / ENGINEERED FEATURES
# ─────────────────────────────────────────────

def compute_derived_features(player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling workload features and ACWR.
    ACWR = acute (7-day) / chronic (28-day) load ratio (Gabbett, 2016).
    """
    df = player_df.copy()

    # Session load proxy (TRIMP-like: distance × intensity index)
    intensity_map = {
        "REST": 0.0, "ACTIVATION": 0.4, "LIGHT_LOAD": 0.5,
        "MEDIUM_LOAD": 0.7, "HIGH_LOAD": 0.85, "MATCH": 1.0
    }
    df["session_load_au"] = (
        df["total_distance_meters"] / 1000
        * df["session_type"].map(intensity_map).fillna(0)
        * df["player_load_score"].clip(lower=0.01).fillna(1)
    ).fillna(0)

    # 7-day rolling average workload
    df["7_day_workload_average"] = (
        df["session_load_au"].rolling(window=7, min_periods=1).mean()
    )

    # Acute load (7 days) and chronic load (28 days)
    acute_load  = df["session_load_au"].rolling(window=7,  min_periods=1).mean()
    chronic_load = df["session_load_au"].rolling(window=28, min_periods=1).mean()
    df["ACWR"] = (acute_load / (chronic_load.replace(0, np.nan))).fillna(1.0).round(3)

    # 7-day HRV trend (slope direction)
    def hrv_trend(series):
        vals = series.dropna()
        if len(vals) < 3:
            return 0.0
        x = np.arange(len(vals))
        slope = np.polyfit(x, vals, 1)[0]
        return round(slope, 3)

    df["7_day_HRV_trend"] = (
        df["HRV_rMSSD"].rolling(window=7, min_periods=3)
                       .apply(hrv_trend, raw=False)
                       .fillna(0)
    )

    # Cumulative fatigue score (exponentially weighted)
    df["cumulative_fatigue_score"] = (
        df["session_load_au"]
          .ewm(span=14, adjust=False)
          .mean()
          .round(3)
    )

    return df


# ─────────────────────────────────────────────
#  INTRODUCE MISSING VALUES (~5%)
# ─────────────────────────────────────────────

def introduce_missing_values(df: pd.DataFrame, missing_rate: float = 0.05,
                              seed: int = 99) -> pd.DataFrame:
    """
    Introduce MCAR (Missing Completely At Random) noise to ~5% of values.
    Excludes identifier and target columns.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    exclude = {"player_id", "date", "session_type", "day_number",
               "fatigue_state", "recovery_readiness", "next_day_fatigue_probability"}

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude]

    n_missing = int(len(df) * len(numeric_cols) * missing_rate)
    rows = rng.integers(0, len(df), n_missing)
    cols = rng.choice(numeric_cols, n_missing)

    for r, c in zip(rows, cols):
        df.iloc[r, df.columns.get_loc(c)] = np.nan

    return df


# ─────────────────────────────────────────────
#  MASTER GENERATION LOOP
# ─────────────────────────────────────────────

def generate_dataset(n_players: int = 9, n_weeks: int = 8,
                     output_dir: str = "../data") -> pd.DataFrame:
    """
    Main orchestration function.
    Generates longitudinal daily records for all players.
    """
    os.makedirs(output_dir, exist_ok=True)
    players_df = create_player_profiles(n_players)
    schedule   = generate_weekly_schedule(n_weeks)
    n_days     = len(schedule)

    start_date = pd.Timestamp("2024-09-02")  # Start of hypothetical season block
    all_records = []

    for _, player in players_df.iterrows():
        pid       = player["player_id"]
        pos       = player["playing_position"]
        hrv_base  = player["hrv_baseline"]
        rec_rate  = player["recovery_rate"]
        fat_sens  = player["fatigue_sensitivity"]

        fatigue         = np.random.uniform(0.10, 0.25)   # starting fatigue
        prev_session    = "REST"
        consec_load     = 0

        for day_idx, stype in enumerate(schedule):
            date = start_date + pd.Timedelta(days=day_idx)

            # Track consecutive load days
            if stype in ["HIGH_LOAD", "MEDIUM_LOAD", "MATCH"]:
                consec_load += 1
            else:
                consec_load = 0

            # External load
            ext = generate_external_load(stype, pos, fatigue, day_idx)

            # Internal load
            intl = generate_internal_load(stype, fatigue, hrv_base, rec_rate)

            # Wellness
            well = generate_wellness(stype, fatigue, prev_session, rec_rate)

            # Update fatigue state
            hrv_val = intl["HRV_rMSSD"] if not np.isnan(intl["HRV_rMSSD"]) else hrv_base
            fatigue = update_fatigue(
                fatigue, stype, hrv_val, hrv_base,
                well["sleep_quality"], well["muscle_soreness"],
                rec_rate, fat_sens
            )

            # Target labels
            hrv_ratio = hrv_val / hrv_base
            fat_state = assign_fatigue_state(fatigue)
            readiness = assign_recovery_readiness(
                fatigue, hrv_ratio, well["sleep_quality"], well["muscle_soreness"]
            )
            rpe_val   = well["RPE"] if not np.isnan(well.get("RPE", np.nan)) else 5.0
            next_prob = next_day_fatigue_prob(
                fatigue, hrv_ratio, well["muscle_soreness"],
                rpe_val, consec_load
            )

            record = {
                "player_id":              pid,
                "date":                   date.date(),
                "day_number":             day_idx + 1,
                "week_number":            (day_idx // 7) + 1,
                "session_type":           stype,
                # Player info
                "age":                    player["age"],
                "playing_position":       pos,
                "baseline_fitness_score": player["baseline_fitness_score"],
                # External load
                **ext,
                # Internal load
                **intl,
                # Wellness
                **well,
                # Fatigue state (hidden continuous for feature engineering)
                "_fatigue_continuous":    round(fatigue, 4),
                # Targets
                "fatigue_state":          fat_state,
                "recovery_readiness":     readiness,
                "next_day_fatigue_probability": next_prob,
            }

            all_records.append(record)
            prev_session = stype

    df = pd.DataFrame(all_records)

    # ── Derived features per player ─────────────────────────────────────
    derived_parts = []
    for pid in df["player_id"].unique():
        part = df[df["player_id"] == pid].copy().reset_index(drop=True)
        part = compute_derived_features(part)
        derived_parts.append(part)
    df = pd.concat(derived_parts, ignore_index=True).sort_values(
        ["player_id", "day_number"]
    ).reset_index(drop=True)

    # ── Introduce missing values ─────────────────────────────────────────
    df = introduce_missing_values(df, missing_rate=0.05)

    # ── Save outputs ─────────────────────────────────────────────────────
    df.to_csv(f"{output_dir}/football_athlete_monitoring.csv", index=False)
    players_df.to_csv(f"{output_dir}/player_profiles.csv", index=False)

    print(f"✅  Dataset generated: {len(df)} records × {len(df.columns)} features")
    print(f"    Players: {df['player_id'].nunique()} | Days: {n_days} | Weeks: {n_weeks}")
    print(f"    Shape: {df.shape}")
    print(f"    Saved → {output_dir}/football_athlete_monitoring.csv")
    return df


# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = generate_dataset(n_players=9, n_weeks=8, output_dir="../data")
    print("\nFeature summary:")
    print(df.describe().round(2))
    print("\nFatigue state distribution:")
    print(df["fatigue_state"].value_counts())
    print("\nRecovery readiness distribution:")
    print(df["recovery_readiness"].value_counts())
