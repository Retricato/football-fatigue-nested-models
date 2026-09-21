"""
config.py — central configuration for the real-data thesis pipeline.

This is the ONLY file you should need to edit by hand.

Day-3 task: run `python 04_fetch_soccermon.py --inspect` and fill in the
right-hand side of SCHEMA_MAP with the actual column names printed.
Every entry marked TODO(day3) is a guess based on the SoccerMon paper's
variable descriptions, not on the archive itself.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────

# repository root: config.py lives in src/, everything else is one level up
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR         = ROOT / "data"
RAW_DIR          = DATA_DIR / "soccermon"
OBJECTIVE_DIR    = RAW_DIR / "objective"

# The unpacked subjective archive. Checked in this order so it works whether
# you dropped `subjective/` in the project root or let 04_fetch_soccermon.py
# unpack it under data/soccermon/.
_SUBJ_CANDIDATES = [ROOT / "subjective", RAW_DIR / "subjective"]
SUBJECTIVE_DIR = next((p for p in _SUBJ_CANDIDATES
                       if (p / "wellness").is_dir()), _SUBJ_CANDIDATES[0])
REAL_DIR         = DATA_DIR / "real"
FIG_DIR          = ROOT / "figures"
RESULTS_DIR      = ROOT / "results"

SYNTHETIC_CSV    = ROOT / "football_athlete_monitoring.csv"
REAL_DAILY_CSV   = REAL_DIR / "soccermon_daily.csv"
REAL_MODEL_CSV   = REAL_DIR / "soccermon_model_ready.csv"

for _d in (DATA_DIR, RAW_DIR, SUBJECTIVE_DIR, OBJECTIVE_DIR,
           REAL_DIR, FIG_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
#  ZENODO SOURCE
#  SoccerMon, DOI 10.5281/zenodo.10033832, CC BY 4.0
# ─────────────────────────────────────────────────────────────

ZENODO_RECORD = "10033832"
ZENODO_BASE   = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

ZENODO_FILES = {
    # name: (filename, size_bytes_approx, md5)
    "subjective": (
        "subjective.zip", 705_800,
        "a3e86aeca611f77c9331a535eae00bf7",
    ),
    # Track B only. Do not download before the Day-18 gate.
    "objective-TeamB-2020": (
        "objective-TeamB-2020.zip", 16_400_000_000,
        "c68f9d77ebaf042ab3e75a0714587c3c",
    ),
    "objective-TeamA-2020": (
        "objective-TeamA-2020.zip", 23_300_000_000,
        "e70a71d0a2e39f939fcd13ec5c73411e",
    ),
    "objective-TeamB-2021": (
        "objective-TeamB-2021.zip", 28_400_000_000,
        "4676436c79d2d0ab3fe1c521a8fd2871",
    ),
    "objective-TeamA-2021": (
        "objective-TeamA-2021.zip", 31_000_000_000,
        "1c0de36af51c13a4fbde463fa808db0e",
    ),
}

CITATION = (
    "Midoglu, C., Boeker, M., Kjaereng Winther, A., Pettersen, S.A., "
    "Johansen, D., Riegler, M., Halvorsen, P., Hicks, S. (2022). SoccerMon "
    "(v1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10033832 "
    "(CC BY 4.0). Described in: Midoglu et al. (2024), Scientific Data 11, 471."
)

# ─────────────────────────────────────────────────────────────
#  ACTUAL SoccerMon SUBJECTIVE LAYOUT  (verified against the archive)
#
#  The archive is NOT one row per player-day. It is:
#
#    subjective/
#      wellness/*.csv        WIDE: rows = dates (dd.mm.yyyy), cols = 50 player
#                            UUIDs. One file per metric.
#      training-load/*.csv   WIDE, same shape. Pre-computed daily metrics.
#      training-load/session.json
#                            LONG: {player_uuid: [{srpe, rpe, duration, date}]}
#                            16,265 individual sessions.
#      injury/injury.csv     LONG: player_name, type (JSON {body_part: severity}),
#                            timestamp
#      illness/illness.csv   LONG: player_name, problems (JSON list), timestamp
#      game-performance/…    LONG: player_name, 3 ratings, timestamp
#
#  50 players (27 Team A, 23 Team B), 731 days: 2020-01-01 to 2021-12-31.
# ─────────────────────────────────────────────────────────────

DATE_FORMAT = "%d.%m.%Y"          # European, throughout the archive

# wide file  ->  column name it becomes after melting
WELLNESS_FILES = {
    "fatigue.csv":        "fatigue_raw",
    "mood.csv":           "mood_raw",
    "readiness.csv":      "readiness",          # 1-10
    "sleep_quality.csv":  "sleep_quality_raw",
    "soreness.csv":       "soreness_raw",
    "stress.csv":         "stress_raw",
    "sleep_duration.csv": "sleep_hours",        # already decimal hours
}

TRAINING_LOAD_FILES = {
    "daily_load.csv":  "daily_load_provided",
    "weekly_load.csv": "weekly_load_provided",
    "acwr.csv":        "acwr_provided",         # capped at 4.0 by the authors
    "atl.csv":         "atl_provided",
    "ctl28.csv":       "ctl28_provided",        # two chronic windows are shipped
    "ctl42.csv":       "ctl42_provided",
    "monotony.csv":    "monotony_provided",
    "strain.csv":      "strain_provided",
}

# ─────────────────────────────────────────────────────────────
#  SCALE POLARITY  —  read this before touching the labels
#
#  Every PMSys wellness item is scored so that HIGHER = BETTER.
#  fatigue 5 means "fresh", not "exhausted"; soreness 5 means "not sore".
#  This is the opposite of the Hooper convention and is easy to get wrong.
#
#  Verified empirically against the archive (correlation with readiness,
#  which is unambiguously higher = more ready to train):
#
#      fatigue        +0.485      soreness       +0.340
#      mood           +0.236      sleep_quality  +0.200
#      stress         +0.195      sleep_duration +0.174
#
#  All positive, so all items point the same way as readiness. Previous-day
#  training load also correlates NEGATIVELY with next-day fatigue (-0.117)
#  and soreness (-0.171): a hard session lowers the score, i.e. a low score
#  means a worse state. Confirmed.
#
#  Re-run this check yourself if you ever change the loader:
#      python 06_build_real_dataset.py --check-polarity
# ─────────────────────────────────────────────────────────────

WELLNESS_HIGHER_IS_BETTER = True

# 1-5 -> 1-10 rescale so the real data is numerically comparable to the
# synthetic schema. The raw 1-5 column is always kept alongside.
# NOTE the semantics: `sleep_quality` and `mood_score` keep the "higher =
# better" direction; `fatigue_score` and `muscle_soreness` are FLIPPED on
# creation so their names mean what they say (high muscle_soreness = sore),
# matching the synthetic dataset's convention.
RESCALE_1_5_TO_1_10 = {
    "sleep_quality_raw": ("sleep_quality",   False),   # (new name, flip?)
    "mood_raw":          ("mood_score",      False),
    "fatigue_raw":       ("fatigue_score",   True),    # -> higher = more fatigued
    "soreness_raw":      ("muscle_soreness", True),    # -> higher = more sore
    "stress_raw":        ("stress_score",    True),    # -> higher = more stressed
}

# Hooper Index components, in the direction where HIGHER = WORSE.
# After the flips above, all four already point that way, so all weights
# are +1. Do not "fix" this back to -1 for sleep_quality without re-reading
# the polarity note -- sleep_quality is deliberately left unflipped and is
# therefore subtracted by the loader via its -1 weight.
HOOPER_COMPONENTS = {
    "fatigue_score":   +1,   # flipped on load: higher = more fatigued
    "muscle_soreness": +1,   # flipped on load: higher = more sore
    "stress_score":    +1,   # flipped on load: higher = more stressed
    "sleep_quality":   -1,   # NOT flipped: higher = slept better -> subtract
}

# ─────────────────────────────────────────────────────────────
#  FEATURE SETS
# ─────────────────────────────────────────────────────────────

# What we can actually build from SoccerMon subjective data (Track A).
REAL_FEATURES = [
    # internal load
    "RPE", "session_duration_minutes", "session_load_au", "n_sessions",
    # wellness
    "sleep_quality", "muscle_soreness", "mood_score", "fatigue_score",
    "stress_score", "sleep_hours", "readiness",
    # derived
    "ACWR", "7_day_workload_average", "28_day_workload_average",
    "cumulative_fatigue_score", "monotony", "strain",
    "hooper_index", "7_day_hooper_trend", "days_since_rest",
    # shipped by the dataset authors -- see note below
    "acwr_provided", "atl_provided", "ctl28_provided", "monotony_provided",
]

# The authors' own ACWR/ATL/CTL/monotony/strain columns are included above.
# Two cautions for Methods:
#   - they are computed over the player's whole record, so treat them as
#     descriptive; our recomputed trailing-window versions are the ones that
#     are safe for prospective prediction.
#   - `acwr_provided` is capped at 4.0 by the authors.
# If you would rather use only your own derived features, drop the four
# `*_provided` entries from REAL_FEATURES and say so.

# Track B adds these, if and only if 10_gps_aggregate.py has run and merged.
# Names match the output of that script exactly.
GPS_FEATURES = [
    "total_distance_meters", "high_speed_running_distance", "sprint_distance",
    "acceleration_count", "deceleration_count",
    "gps_active_minutes", "peak_speed_ms", "n_gps_sessions",
]

# Present in synthetic only. Declared as a limitation for the real data.
#
# Note on heart rate: the objective archive HAS a `heart_rate` column, but it
# is either 0 or a constant placeholder (82 in every file inspected). It is
# not a real HR trace. Do not build features on it -- the "no HRV, no heart
# rate" limitation stands even with the GPS data in hand.
UNAVAILABLE_IN_SOCCERMON = [
    "HRV_rMSSD", "resting_heart_rate", "average_training_hr", "max_training_hr",
    "7_day_HRV_trend", "recovery_index", "age", "baseline_fitness_score",
    "playing_position", "player_load_score",
]

# Shared subset used for the synthetic <-> real transfer experiment (09).
TRANSFER_FEATURES = [
    "RPE", "session_duration_minutes", "session_load_au",
    "sleep_quality", "muscle_soreness", "mood_score",
    "ACWR", "7_day_workload_average", "cumulative_fatigue_score",
]

# ─────────────────────────────────────────────────────────────
#  LABEL CONFIGURATION
# ─────────────────────────────────────────────────────────────

LABEL_CONFIGS = {
    # Primary. Per-player rolling z-threshold on the Hooper Index.
    "A_hooper_1.0sd": {
        "source": "hooper_index", "method": "rolling_z",
        "window": 28, "threshold": 1.0, "min_periods": 14,
    },
    # Secondary. Bottom-tertile readiness, within player.
    "B_readiness_tertile": {
        "source": "readiness", "method": "within_player_quantile",
        "quantile": 0.33, "direction": "below",
    },
    # Sensitivity variants.
    "C_hooper_1.5sd": {
        "source": "hooper_index", "method": "rolling_z",
        "window": 28, "threshold": 1.5, "min_periods": 14,
    },
    "D_hooper_0.5sd": {
        "source": "hooper_index", "method": "rolling_z",
        "window": 28, "threshold": 0.5, "min_periods": 14,
    },
}

PRIMARY_LABEL = "A_hooper_1.0sd"

# Predict the label at t+1 from features at t. Never change this to 0
# without a very good reason -- same-day prediction is not useful to a
# coach and makes the wellness features near-tautological.
LABEL_HORIZON_DAYS = 1

# ─────────────────────────────────────────────────────────────
#  COHORT SELECTION  (pre-register this before looking at results)
# ─────────────────────────────────────────────────────────────

COHORT_RULES = {
    "min_wellness_days":   150,
    "min_load_sessions":   100,
    "min_positive_labels":  10,   # needed for LOPO to be meaningful
}

# ─────────────────────────────────────────────────────────────
#  MODELLING
# ─────────────────────────────────────────────────────────────

RANDOM_SEED = 42

CV_CONFIG = {
    "skf_splits":        5,     # optimistic, kept for synthetic comparability
    "grouped_splits":    5,     # StratifiedGroupKFold on player_id -- primary
    "lopo":              True,  # leave-one-player-out -- headline metric
}

LSTM_CONFIG = {
    "seq_len":        7,
    "max_gap_days":   2,    # do not build a window across a longer gap
    "units":          64,
    "dropout":        0.3,
    "epochs":         60,
    "batch_size":     32,
    "patience":       10,
    "class_weight":   True,
}

# Report these first. Accuracy is deliberately NOT primary -- with
# 10-20% positives it is uninformative.
PRIMARY_METRICS = ["pr_auc", "balanced_accuracy", "f1", "roc_auc"]
