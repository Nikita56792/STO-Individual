"""
Main training script for the LightGBM model.

Uses temporal split with absolute date threshold to ensure methodologically
correct validation without data leakage from future timestamps.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
from .temporal_split import get_split_date_from_ratio, temporal_split_by_date


def train() -> None:
    """Runs the model training pipeline with temporal split.

    Loads prepared data from data/processed/, performs temporal split based on
    absolute date threshold, computes aggregate features on train split only,
    and trains a single LightGBM model. This ensures methodologically correct
    validation without data leakage from future timestamps.

    Note: Data must be prepared first using prepare_data.py
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

    # Check for timestamp column
    if constants.COL_TIMESTAMP not in train_set.columns:
        raise ValueError(
            f"Timestamp column '{constants.COL_TIMESTAMP}' not found in train set. "
            "Make sure data was prepared with timestamp preserved."
        )

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(train_set[constants.COL_TIMESTAMP]):
        train_set[constants.COL_TIMESTAMP] = pd.to_datetime(train_set[constants.COL_TIMESTAMP])

    # Perform temporal split
    print(f"\nPerforming temporal split with ratio {config.TEMPORAL_SPLIT_RATIO}...")
    split_date = get_split_date_from_ratio(train_set, config.TEMPORAL_SPLIT_RATIO, constants.COL_TIMESTAMP)
    print(f"Split date: {split_date}")

    train_mask, val_mask = temporal_split_by_date(train_set, split_date, constants.COL_TIMESTAMP)

    # Split data
    train_split = train_set[train_mask].copy()
    val_split = train_set[val_mask].copy()

    print(f"Train split: {len(train_split):,} rows")
    print(f"Validation split: {len(val_split):,} rows")

    # Verify temporal correctness
    max_train_timestamp = train_split[constants.COL_TIMESTAMP].max()
    min_val_timestamp = val_split[constants.COL_TIMESTAMP].min()
    print(f"Max train timestamp: {max_train_timestamp}")
    print(f"Min validation timestamp: {min_val_timestamp}")

    if min_val_timestamp <= max_train_timestamp:
        raise ValueError(
            f"Temporal split validation failed: min validation timestamp ({min_val_timestamp}) "
            f"is not greater than max train timestamp ({max_train_timestamp})."
        )
    print("✅ Temporal split validation passed: all validation timestamps are after train timestamps")

    print("\nBuilding latent factors from train split...")
    user_factors, book_factors = compute_latent_factors(train_split, n_components=config.LATENT_DIM)
    user_factors_imp, book_factors_imp = compute_implicit_latent_factors(n_components=config.LATENT_DIM)

    # Compute aggregate features on train split only (to prevent data leakage)
    print("\nComputing aggregate features on train split only...")
    train_split_with_agg = add_aggregate_features(train_split.copy(), train_split)
    val_split_with_agg = add_aggregate_features(val_split.copy(), train_split)  # Use train_split for aggregates!
    train_split_with_agg = add_latent_features(train_split_with_agg, user_factors, book_factors)
    val_split_with_agg = add_latent_features(val_split_with_agg, user_factors, book_factors)
    train_split_with_agg = add_latent_features(train_split_with_agg, user_factors_imp, book_factors_imp)
    val_split_with_agg = add_latent_features(val_split_with_agg, user_factors_imp, book_factors_imp)
    train_split_with_agg = add_latent_dot_products(train_split_with_agg)
    val_split_with_agg = add_latent_dot_products(val_split_with_agg)

    print("\nFitting Surprise models for additional collaborative signals...")
    surprise_models = train_surprise_models(train_split)
    train_split_with_agg = apply_surprise_models(train_split_with_agg, surprise_models)
    val_split_with_agg = apply_surprise_models(val_split_with_agg, surprise_models)

    print("Fitting ALS models for implicit and explicit patterns...")
    als_models = train_als_models(train_split)
    train_split_with_agg = apply_als_models(train_split_with_agg, als_models)
    val_split_with_agg = apply_als_models(val_split_with_agg, als_models)

    # Handle missing values (use train_split for fill values)
    print("Handling missing values...")
    train_split_final = handle_missing_values(train_split_with_agg, train_split)
    val_split_final = handle_missing_values(val_split_with_agg, train_split)

    # Define features (X) and target (y)
    # Exclude timestamp, source, target, prediction columns
    exclude_cols = [
        constants.COL_SOURCE,
        config.TARGET,
        constants.COL_PREDICTION,
        constants.COL_TIMESTAMP,
    ]
    features = [col for col in train_split_final.columns if col not in exclude_cols]

    # Exclude any remaining object columns that are not model features
    non_feature_object_cols = train_split_final[features].select_dtypes(include=["object"]).columns.tolist()
    features = [f for f in features if f not in non_feature_object_cols]

    X_train = train_split_final[features]
    y_train = train_split_final[config.TARGET]
    X_val = val_split_final[features]
    y_val = val_split_final[config.TARGET]

    print(f"Training features: {len(features)}")

    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Train multiple LightGBM seeds for variance reduction
    print("\nTraining LightGBM models (multi-seed ensemble)...")
    lgb_models: list[lgb.LGBMRegressor] = []
    lgb_val_preds_list = []
    for seed_idx, seed in enumerate(config.LGB_SEEDS):
        params = config.LGB_PARAMS.copy()
        params["seed"] = seed
        model = lgb.LGBMRegressor(**params)

        fit_params = config.LGB_FIT_PARAMS.copy()
        fit_params["callbacks"] = [lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=False)]

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=fit_params["eval_metric"],
            callbacks=fit_params["callbacks"],
        )

        preds = model.predict(X_val)
        lgb_models.append(model)
        lgb_val_preds_list.append(preds)
        model_path = config.MODEL_DIR / constants.LGB_MODEL_PATTERN.format(seed=seed)
        model.booster_.save_model(str(model_path))
        if seed_idx == 0:
            # Backward compatibility: save first seed to the default filename
            default_path = config.MODEL_DIR / config.MODEL_FILENAME
            model.booster_.save_model(str(default_path))
    lgb_val_preds = np.mean(lgb_val_preds_list, axis=0)

    # Train CatBoost (works well with categorical features)
    print("\nTraining CatBoost model...")
    cat_features_base = [f for f in features if not f.startswith("tfidf_") and not f.startswith("bert_")]
    cat_feature_cols = [
        c for c in cat_features_base if c in config.CAT_FEATURES and X_train[c].dtype.name in ("category", "object")
    ]
    cat_features_idx = [cat_features_base.index(c) for c in cat_feature_cols]

    X_train_cat = X_train[cat_features_base].copy()
    X_val_cat = X_val[cat_features_base].copy()
    for col in cat_feature_cols:
        X_train_cat[col] = X_train_cat[col].astype(str)
        X_val_cat[col] = X_val_cat[col].astype(str)

    train_pool = Pool(X_train_cat, label=y_train, cat_features=cat_features_idx)
    val_pool = Pool(X_val_cat, label=y_val, cat_features=cat_features_idx)

    cat_model = CatBoostRegressor(**config.CATBOOST_PARAMS, verbose=False)
    cat_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    cat_val_preds = cat_model.predict(val_pool)

    # Evaluate ensemble with simple weight search on validation
    def _score(preds: np.ndarray) -> tuple[float, float, float]:
        rmse_v = np.sqrt(mean_squared_error(y_val, preds))
        mae_v = mean_absolute_error(y_val, preds)
        score_v = 1 - 0.5 * ((rmse_v / 10) + (mae_v / 10))
        return score_v, rmse_v, mae_v

    best_score, best_rmse, best_mae, best_w_lgb = -np.inf, None, None, config.ENSEMBLE_WEIGHTS["lgb"]
    for w_cat_try in np.linspace(0, 1, 21):
        w_lgb_try = 1 - w_cat_try
        preds_try = w_lgb_try * lgb_val_preds + w_cat_try * cat_val_preds
        score_v, rmse_v, mae_v = _score(preds_try)
        if score_v > best_score:
            best_score, best_rmse, best_mae, best_w_lgb = score_v, rmse_v, mae_v, w_lgb_try

    w_lgb = best_w_lgb
    w_cat = 1 - w_lgb
    val_preds = w_lgb * lgb_val_preds + w_cat * cat_val_preds

    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    mae = mean_absolute_error(y_val, val_preds)
    print(f"\nValidation (ensemble) RMSE: {rmse:.4f}, MAE: {mae:.4f} | w_lgb={w_lgb:.2f} w_cat={w_cat:.2f} | best_score={best_score:.4f}")

    calibration = {
        "w_lgb": float(w_lgb),
        "w_cat": float(w_cat),
        "best_score": float(best_score),
        "rmse": float(rmse),
        "mae": float(mae),
        "split_date": split_date.isoformat(),
    }
    calibration_path = config.MODEL_DIR / constants.CALIBRATION_FILENAME
    with open(calibration_path, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)
    print(f"Saved ensemble weights to {calibration_path}")

    # Save models
    model_path = config.MODEL_DIR / config.MODEL_FILENAME
    if lgb_models:
        lgb_models[0].booster_.save_model(str(model_path))
    print(f"LightGBM models saved to {config.MODEL_DIR} (seeds={config.LGB_SEEDS})")

    cat_path = config.MODEL_DIR / config.CAT_MODEL_FILENAME
    cat_model.save_model(cat_path)
    print(f"CatBoost model saved to {cat_path}")

    if config.TRAIN_ON_FULL:
        print("\nRetraining both models on full training data (no holdout)...")
        user_factors_full, book_factors_full = compute_latent_factors(train_set, n_components=config.LATENT_DIM)
        user_factors_imp_full, book_factors_imp_full = compute_implicit_latent_factors(
            n_components=config.LATENT_DIM
        )
        full_train_with_agg = add_aggregate_features(train_set.copy(), train_set)
        full_train_with_agg = add_latent_features(full_train_with_agg, user_factors_full, book_factors_full)
        full_train_with_agg = add_latent_features(full_train_with_agg, user_factors_imp_full, book_factors_imp_full)
        full_train_with_agg = add_latent_dot_products(full_train_with_agg)
        surprise_models_full = train_surprise_models(train_set)
        full_train_with_agg = apply_surprise_models(full_train_with_agg, surprise_models_full)
        als_models_full = train_als_models(train_set)
        full_train_with_agg = apply_als_models(full_train_with_agg, als_models_full)
        full_train_final = handle_missing_values(full_train_with_agg, train_set)

        X_full = full_train_final[features]
        y_full = full_train_final[config.TARGET]

        for seed_idx, seed in enumerate(config.LGB_SEEDS):
            params = config.LGB_PARAMS.copy()
            params["seed"] = seed
            lgb_full = lgb.LGBMRegressor(**params)
            lgb_full.fit(X_full, y_full)

            full_path = config.MODEL_DIR / constants.LGB_MODEL_PATTERN.format(seed=seed)
            lgb_full.booster_.save_model(str(full_path))
            if seed_idx == 0:
                lgb_full.booster_.save_model(str(model_path))
        print(f"Full LightGBM models saved to {config.MODEL_DIR} (seeds={config.LGB_SEEDS})")

        X_full_cat = X_full[cat_features_base]
        cat_full_pool = Pool(X_full_cat, label=y_full, cat_features=cat_features_idx)
        cat_full = CatBoostRegressor(**config.CATBOOST_PARAMS, verbose=False)
        cat_full.fit(cat_full_pool)
        cat_full.save_model(cat_path)
        print(f"Full CatBoost model saved to {cat_path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    train()
