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
USE_SURPRISE = os.getenv("USE_SURPRISE", "1") != "0"
USE_IMPLICIT = os.getenv("USE_IMPLICIT", "1") != "0"
LATENT_DIM = int(os.getenv("LATENT_DIM", "96"))

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
BERT_MAX_LENGTH = 320
BERT_EMBEDDING_DIM = 1024
BERT_SVD_COMPONENTS = int(os.getenv("BERT_SVD_COMPONENTS", "384"))
BERT_DEVICE = _detect_bert_device()
# Limit GPU memory usage to 50% to prevent overheating and OOM errors
BERT_GPU_MEMORY_FRACTION = float(os.getenv("BERT_GPU_MEMORY_FRACTION", "0.75"))

SURPRISE_SVD_PARAMS = {
    "n_factors": 128,
    "n_epochs": 40,
    "lr_all": 0.006,
    "reg_all": 0.08,
    "random_state": RANDOM_STATE,
}
SURPRISE_SVDPP_PARAMS = {
    "n_factors": 96,
    "n_epochs": 35,
    "lr_all": 0.005,
    "reg_all": 0.09,
    "random_state": RANDOM_STATE,
}
SURPRISE_KNN_PARAMS = {
    "k": 80,
    "min_k": 3,
    "sim_options": {"name": "pearson_baseline", "user_based": False},
}

IMPLICIT_ALS_PARAMS = {
    "factors": 96,
    "regularization": 0.08,
    "alpha": 15.0,
    "iterations": 30,
    "random_state": RANDOM_STATE,
}


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
    "learning_rate": 0.03,
    "depth": 9,
    "l2_leaf_reg": 5.0,
    "min_data_in_leaf": 18,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.88,
    "iterations": 3200,
    "od_type": "Iter",
    "od_wait": 80,
    "random_seed": RANDOM_STATE,
    "task_type": "CPU",
    "thread_count": NUM_THREADS,
}

# Ensemble weights
ENSEMBLE_WEIGHTS = {"lgb": 0.8, "cat": 0.2}

# --- MODEL PARAMETERS ---
LGB_PARAMS = {
    "objective": "rmse",
    "metric": "rmse",
    "n_estimators": 12000,
    "learning_rate": 0.0125,
    "feature_fraction": 0.92,
    "bagging_fraction": 0.86,
    "bagging_freq": 1,
    "lambda_l1": 0.12,
    "lambda_l2": 0.18,
    "num_leaves": 320,
    "min_child_samples": 16,
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
LGB_SEEDS = [42, 2025, 7, 1337, 31415]
