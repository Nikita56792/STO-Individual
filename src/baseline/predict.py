"""
Inference script to generate predictions for the test set.

Computes aggregate features on all train data and applies them to test set,
then generates predictions using the trained model.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import json

from . import config, constants
from .features import (
    add_aggregate_features,
    add_latent_features,
    add_latent_dot_products,
    apply_als_models,
    apply_surprise_models,
    compute_implicit_latent_factors,
    compute_latent_factors,
    handle_missing_values,
    train_als_models,
    train_surprise_models,
)


def predict() -> None:
    """Generates and saves predictions for the test set.

    This script loads prepared data from data/processed/, computes aggregate features
    on all train data, applies them to test set, and generates predictions using
    the trained model.

    Note: Data must be prepared first using prepare_data.py, and model must be trained
    using train.py
    """
    # Load prepared data
    processed_path = config.PROCESSED_DATA_DIR / constants.PROCESSED_DATA_FILENAME

    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {processed_path}. "
            "Please run 'poetry run python -m src.baseline.prepare_data' first."
        )

    print(f"Loading prepared data from {processed_path}...")
    featured_df = pd.read_parquet(processed_path, engine="pyarrow")
    print(f"Loaded {len(featured_df):,} rows with {len(featured_df.columns)} features")

    # Separate train and test sets
    train_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()
    test_set = featured_df[featured_df[constants.COL_SOURCE] == constants.VAL_SOURCE_TEST].copy()

    print(f"Train set: {len(train_set):,} rows")
    print(f"Test set: {len(test_set):,} rows")

    print("\nBuilding latent factors from full train data...")
    user_factors, book_factors = compute_latent_factors(train_set, n_components=config.LATENT_DIM)
    user_factors_imp, book_factors_imp = compute_implicit_latent_factors(n_components=config.LATENT_DIM)

    # Compute aggregate features on ALL train data (to use for test predictions)
    print("\nComputing aggregate features on all train data...")
    test_set_with_agg = add_aggregate_features(test_set.copy(), train_set)
    test_set_with_agg = add_latent_features(test_set_with_agg, user_factors, book_factors)
    test_set_with_agg = add_latent_features(test_set_with_agg, user_factors_imp, book_factors_imp)
    test_set_with_agg = add_latent_dot_products(test_set_with_agg)
    print("Fitting collaborative models on full train data for inference...")
    surprise_models = train_surprise_models(train_set)
    test_set_with_agg = apply_surprise_models(test_set_with_agg, surprise_models)
    als_models = train_als_models(train_set)
    test_set_with_agg = apply_als_models(test_set_with_agg, als_models)

    # Handle missing values (use train_set for fill values)
    print("Handling missing values...")
    test_set_final = handle_missing_values(test_set_with_agg, train_set)

    # Define features (exclude source, target, prediction, timestamp columns)
    exclude_cols = [
        constants.COL_SOURCE,
        config.TARGET,
        constants.COL_PREDICTION,
        constants.COL_TIMESTAMP,
    ]
    features = [col for col in test_set_final.columns if col not in exclude_cols]

    # Exclude any remaining object columns that are not model features
    non_feature_object_cols = test_set_final[features].select_dtypes(include=["object"]).columns.tolist()
    features = [f for f in features if f not in non_feature_object_cols]

    X_test = test_set_final[features]
    print(f"Prediction features: {len(features)}")

    # Load trained models
    lgb_path = config.MODEL_DIR / config.MODEL_FILENAME
    cat_path = config.MODEL_DIR / config.CAT_MODEL_FILENAME
    if not lgb_path.exists() or not cat_path.exists():
        raise FileNotFoundError(
            f"Models not found at {lgb_path} and/or {cat_path}. "
            "Please run 'poetry run python -m src.baseline.train' first."
        )

    print(f"\nLoading models from {lgb_path} and {cat_path}...")
    lgb_model = lgb.Booster(model_file=str(lgb_path))
    cat_model = CatBoostRegressor()
    cat_model.load_model(cat_path)

    # Generate predictions
    print("Generating predictions (ensemble)...")
    lgb_preds_list = []
    for seed in config.LGB_SEEDS:
        path = config.MODEL_DIR / constants.LGB_MODEL_PATTERN.format(seed=seed)
        if not path.exists():
            continue
        booster = lgb.Booster(model_file=str(path))
        lgb_preds_list.append(booster.predict(X_test))
    if not lgb_preds_list:
        lgb_preds_list.append(lgb_model.predict(X_test))
    lgb_preds = np.mean(lgb_preds_list, axis=0)

    cat_features_base = [f for f in features if not f.startswith("tfidf_") and not f.startswith("bert_")]
    cat_feature_cols = [
        c for c in cat_features_base if c in config.CAT_FEATURES and X_test[c].dtype.name in ("category", "object")
    ]
    cat_features_idx = [cat_features_base.index(c) for c in cat_feature_cols]

    X_test_cat = X_test[cat_features_base].copy()
    for col in cat_feature_cols:
        X_test_cat[col] = X_test_cat[col].astype(str)

    cat_pool = Pool(X_test_cat, cat_features=cat_features_idx)
    cat_preds = cat_model.predict(cat_pool)

    w_lgb = config.ENSEMBLE_WEIGHTS["lgb"]
    w_cat = config.ENSEMBLE_WEIGHTS["cat"]
    calibration_path = config.MODEL_DIR / constants.CALIBRATION_FILENAME
    if calibration_path.exists():
        with open(calibration_path, "r", encoding="utf-8") as f:
            calibration = json.load(f)
        w_lgb = float(calibration.get("w_lgb", w_lgb))
        w_cat = float(calibration.get("w_cat", w_cat))
        norm = w_lgb + w_cat
        if norm > 0:
            w_lgb, w_cat = w_lgb / norm, w_cat / norm
        print(f"Using calibrated weights from {calibration_path}: w_lgb={w_lgb:.2f}, w_cat={w_cat:.2f}")
    else:
        print(f"Calibration file not found at {calibration_path}, using config weights w_lgb={w_lgb:.2f} w_cat={w_cat:.2f}")

    test_preds = w_lgb * lgb_preds + w_cat * cat_preds
    clipped_preds = np.clip(test_preds, constants.PREDICTION_MIN_VALUE, constants.PREDICTION_MAX_VALUE)

    # Create submission file
    submission_df = test_set[[constants.COL_USER_ID, constants.COL_BOOK_ID]].copy()
    submission_df[constants.COL_PREDICTION] = clipped_preds

    # Ensure submission directory exists
    config.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = config.SUBMISSION_DIR / constants.SUBMISSION_FILENAME

    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission file created at: {submission_path}")
    print(f"Predictions: min={clipped_preds.min():.4f}, max={clipped_preds.max():.4f}, mean={clipped_preds.mean():.4f}")


if __name__ == "__main__":
    predict()
