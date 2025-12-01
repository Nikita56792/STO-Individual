"""
Project-wide constants.

This module defines constants that are part of the data schema or project
structure but are not intended to be tuned as hyperparameters.
"""

# --- FILENAMES ---
TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"
USER_DATA_FILENAME = "users.csv"
BOOK_DATA_FILENAME = "books.csv"
BOOK_GENRES_FILENAME = "book_genres.csv"
GENRES_FILENAME = "genres.csv"
BOOK_DESCRIPTIONS_FILENAME = "book_descriptions.csv"
SUBMISSION_FILENAME = "submission.csv"
TFIDF_VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"
BERT_EMBEDDINGS_FILENAME = "bert_embeddings.pkl"
BERT_SVD_FILENAME = "bert_svd.pkl"
BERT_MODEL_NAME = "DeepPavlov/rubert-base-cased"
PROCESSED_DATA_FILENAME = "processed_features.parquet"
LGB_MODEL_FILENAME = "lgb_model.txt"
LGB_MODEL_PATTERN = "lgb_model_{seed}.txt"
CAT_MODEL_FILENAME = "cat_model.cbm"
CALIBRATION_FILENAME = "calibration.json"

# --- COLUMN NAMES ---
# Main columns
COL_USER_ID = "user_id"
COL_BOOK_ID = "book_id"
COL_TARGET = "rating"
COL_SOURCE = "source"
COL_PREDICTION = "rating_predict"
COL_HAS_READ = "has_read"
COL_TIMESTAMP = "timestamp"

# Feature columns (newly created)
F_USER_MEAN_RATING = "user_mean_rating"
F_USER_RATINGS_COUNT = "user_ratings_count"
F_USER_RATINGS_STD = "user_ratings_std"
F_USER_RATINGS_MEDIAN = "user_ratings_median"
F_USER_LAST_RATING = "user_last_rating"
F_USER_RECENCY_DAYS = "user_recency_days"
F_USER_ACTIVITY_DAYS = "user_activity_days"
F_USER_BIAS = "user_bias"
F_BOOK_MEAN_RATING = "book_mean_rating"
F_BOOK_RATINGS_COUNT = "book_ratings_count"
F_BOOK_RATINGS_STD = "book_ratings_std"
F_BOOK_RATINGS_MEDIAN = "book_ratings_median"
F_BOOK_LAST_RATING = "book_last_rating"
F_BOOK_RECENCY_DAYS = "book_recency_days"
F_BOOK_BIAS = "book_bias"
F_BOOK_WISHLIST_COUNT = "book_wishlist_count"
F_AUTHOR_MEAN_RATING = "author_mean_rating"
F_AUTHOR_RATINGS_COUNT = "author_ratings_count"
F_AUTHOR_BIAS = "author_bias"
F_LANGUAGE_MEAN_RATING = "language_mean_rating"
F_PUBLISHER_MEAN_RATING = "publisher_mean_rating"
F_BOOK_GENRES_COUNT = "book_genres_count"
COL_AGE_BUCKET = "age_bucket"
COL_BOOK_AGE = "book_age"
F_BOOK_AGE_AT_TS = "book_age_at_ts"
F_TS_YEAR = "ts_year"
F_TS_MONTH = "ts_month"
F_TS_WEEKDAY = "ts_weekday"
F_TS_HOUR = "ts_hour"
F_TS_MONTH_MEAN_RATING = "ts_month_mean_rating"
F_USER_TOTAL_INTERACTIONS = "user_total_interactions"
F_USER_WISHLIST_COUNT = "user_wishlist_count"
F_USER_READ_COUNT_ALL = "user_read_count_all"
F_USER_READ_RATIO = "user_read_ratio"

# Metadata columns from raw data
COL_GENDER = "gender"
COL_AGE = "age"
COL_AUTHOR_ID = "author_id"
COL_PUBLICATION_YEAR = "publication_year"
COL_LANGUAGE = "language"
COL_PUBLISHER = "publisher"
COL_AVG_RATING = "avg_rating"
COL_GENRE_ID = "genre_id"
COL_MAIN_GENRE = "main_genre_id"
COL_DESCRIPTION = "description"

# Genre/user-author enriched features
F_GENRE_MEAN_RATING = "genre_mean_rating"
F_GENRE_RATINGS_COUNT = "genre_ratings_count"
F_GENRE_BIAS = "genre_bias"
F_USER_GENRE_MEAN_RATING = "user_genre_mean_rating"
F_USER_GENRE_RATINGS_COUNT = "user_genre_ratings_count"
F_USER_AUTHOR_MEAN_RATING = "user_author_mean_rating"
F_USER_AUTHOR_RATINGS_COUNT = "user_author_ratings_count"
F_AGE_BUCKET_MEAN_RATING = "age_bucket_mean_rating"
F_PUBLICATION_DECADE_MEAN_RATING = "publication_decade_mean_rating"


# --- VALUES ---
VAL_SOURCE_TRAIN = "train"
VAL_SOURCE_TEST = "test"

# --- MAGIC NUMBERS ---
MISSING_CAT_VALUE = "-1"
MISSING_NUM_VALUE = -1
PREDICTION_MIN_VALUE = 0
PREDICTION_MAX_VALUE = 10
