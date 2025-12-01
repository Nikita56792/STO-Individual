"""
Configuration file for the NTO ML competition baseline.
"""

import os
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from . import constants


def _detect_bert_device() -> str:
    """Choose the best available device with optional override."""
    env_device = os.getenv("BERT_DEVICE")
    if env_device:
        return env_device

    if torch:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


# --- DIRECTORIES ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"


# --- PARAMETERS ---
N_SPLITS = 5  # Deprecated: kept for backwards compatibility, not used in temporal split
RANDOM_STATE = 42
TARGET = constants.COL_TARGET  # Alias for consistency

# --- PERFORMANCE CONFIG ---
NUM_THREADS = int(os.getenv("NUM_THREADS", os.cpu_count() or 8))
TORCH_NUM_THREADS = int(os.getenv("TORCH_NUM_THREADS", NUM_THREADS))

# --- FEATURE TOGGLES ---
# These flags can be overridden via environment variables for faster experiments
USE_TFIDF = os.getenv("USE_TFIDF", "1") != "0"
USE_BERT = os.getenv("USE_BERT", "1") != "0"

# --- TEMPORAL SPLIT CONFIG ---
# Ratio of data to use for training (0 < TEMPORAL_SPLIT_RATIO < 1)
# 0.8 means 80% of data points (by timestamp) go to train, 20% to validation
TEMPORAL_SPLIT_RATIO = 0.92
TRAIN_ON_FULL = os.getenv("TRAIN_ON_FULL", "0") != "0"

# --- TRAINING CONFIG ---
EARLY_STOPPING_ROUNDS = 300
MODEL_FILENAME_PATTERN = "lgb_fold_{fold}.txt"  # Deprecated: kept for backwards compatibility
MODEL_FILENAME = constants.LGB_MODEL_FILENAME  # Single model filename for temporal split
CAT_MODEL_FILENAME = constants.CAT_MODEL_FILENAME

# --- TF-IDF PARAMETERS ---
TFIDF_MAX_FEATURES = 300
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_N_JOBS = NUM_THREADS

# --- BERT PARAMETERS ---
BERT_MODEL_NAME = constants.BERT_MODEL_NAME
BERT_BATCH_SIZE = int(os.getenv("BERT_BATCH_SIZE", "16"))
BERT_MAX_LENGTH = 512
BERT_EMBEDDING_DIM = 768
BERT_SVD_COMPONENTS = int(os.getenv("BERT_SVD_COMPONENTS", "256"))
BERT_DEVICE = _detect_bert_device()
# Limit GPU memory usage to 50% to prevent overheating and OOM errors
BERT_GPU_MEMORY_FRACTION = float(os.getenv("BERT_GPU_MEMORY_FRACTION", "0.75"))


# --- FEATURES ---
CAT_FEATURES = [
    constants.COL_USER_ID,
    constants.COL_BOOK_ID,
    constants.COL_GENDER,
    constants.COL_AGE,
    constants.COL_AGE_BUCKET,
    constants.COL_AUTHOR_ID,
    constants.COL_PUBLICATION_YEAR,
    constants.COL_LANGUAGE,
    constants.COL_MAIN_GENRE,
    constants.COL_PUBLISHER,
    constants.F_TS_MONTH,
    constants.F_TS_WEEKDAY,
    constants.F_TS_HOUR,
]

# --- CATBOOST PARAMETERS ---
CATBOOST_PARAMS = {
    "loss_function": "RMSE",
    "learning_rate": 0.035,
    "depth": 8,
    "l2_leaf_reg": 4.0,
    "min_data_in_leaf": 20,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.85,
    "iterations": 2500,
    "od_type": "Iter",
    "od_wait": 60,
    "random_seed": RANDOM_STATE,
    "task_type": "CPU",
    "thread_count": NUM_THREADS,
}

# Ensemble weights
ENSEMBLE_WEIGHTS = {"lgb": 1.0, "cat": 0.0}

# --- MODEL PARAMETERS ---
LGB_PARAMS = {
    "objective": "rmse",
    "metric": "rmse",
    "n_estimators": 6500,
    "learning_rate": 0.02,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l1": 0.2,
    "lambda_l2": 0.2,
    "num_leaves": 192,
    "min_child_samples": 20,
    "min_split_gain": 0.0,
    "max_depth": -1,
    "boost_from_average": True,
    "verbose": -1,
    "n_jobs": NUM_THREADS,
    "seed": RANDOM_STATE,
    "boosting_type": "gbdt",
}

# LightGBM's fit method allows for a list of callbacks, including early stopping.
# To use it, we need to specify parameters for the early stopping callback.
LGB_FIT_PARAMS = {
    "eval_metric": "rmse",
    "callbacks": [],  # Placeholder for early stopping callback
}

# Train multiple LightGBM seeds for variance reduction
LGB_SEEDS = [42, 2025, 7]
