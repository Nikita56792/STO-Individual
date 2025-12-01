"""
Feature engineering script.
"""

import time

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from . import config, constants


def compute_latent_factors(
    train_df: pd.DataFrame, n_components: int = 32
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build lightweight user/book latent factors via truncated SVD on rating matrix."""
    interactions = train_df[[constants.COL_USER_ID, constants.COL_BOOK_ID, config.TARGET]].dropna()
    if interactions.empty:
        return pd.DataFrame(), pd.DataFrame()

    user_codes, user_uniques = pd.factorize(interactions[constants.COL_USER_ID])
    book_codes, book_uniques = pd.factorize(interactions[constants.COL_BOOK_ID])

    max_components = min(n_components, len(user_uniques) - 1, len(book_uniques) - 1)
    if max_components < 2:
        return pd.DataFrame(), pd.DataFrame()

    ratings = interactions[config.TARGET].to_numpy()
    matrix = sparse.coo_matrix((ratings, (user_codes, book_codes)), shape=(len(user_uniques), len(book_uniques)))

    svd = TruncatedSVD(n_components=max_components, random_state=config.RANDOM_STATE)
    user_latent = svd.fit_transform(matrix)
    book_latent = svd.components_.T

    user_cols = [f"lf_user_{i}" for i in range(user_latent.shape[1])]
    book_cols = [f"lf_book_{i}" for i in range(book_latent.shape[1])]

    user_factors = pd.DataFrame(user_latent, columns=user_cols)
    user_factors[constants.COL_USER_ID] = pd.Series(user_uniques, index=user_factors.index).astype(np.int64)

    book_factors = pd.DataFrame(book_latent, columns=book_cols)
    book_factors[constants.COL_BOOK_ID] = pd.Series(book_uniques, index=book_factors.index).astype(np.int64)

    return user_factors, book_factors


def add_latent_features(
    df: pd.DataFrame, user_factors: pd.DataFrame, book_factors: pd.DataFrame
) -> pd.DataFrame:
    """Adds latent factor columns to the provided dataframe."""
    if user_factors.empty or book_factors.empty:
        return df

    df = df.merge(user_factors, on=constants.COL_USER_ID, how="left")
    df = df.merge(book_factors, on=constants.COL_BOOK_ID, how="left")
    return df


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds simple deterministic features that do not depend on the target."""
    print("Adding basic metadata features...")

    # Age buckets help the model capture non-linear effects of age
    age_bins = [0, 18, 25, 35, 50, 120]
    age_labels = ["<18", "18-24", "25-34", "35-49", "50+"]
    df[constants.COL_AGE_BUCKET] = pd.cut(df[constants.COL_AGE], bins=age_bins, labels=age_labels, right=False)

    # Book age: newer vs. older publications
    current_year = pd.Timestamp.now().year
    df[constants.COL_BOOK_AGE] = current_year - df[constants.COL_PUBLICATION_YEAR]
    df[constants.F_BOOK_AGE_AT_TS] = np.nan

    # Timestamp-based periodic features (train rows only; test will be NaN -> filled)
    if constants.COL_TIMESTAMP in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[constants.COL_TIMESTAMP]):
            df[constants.COL_TIMESTAMP] = pd.to_datetime(df[constants.COL_TIMESTAMP], errors="coerce")
        df[constants.F_TS_YEAR] = df[constants.COL_TIMESTAMP].dt.year
        df[constants.F_TS_MONTH] = df[constants.COL_TIMESTAMP].dt.month
        df[constants.F_TS_WEEKDAY] = df[constants.COL_TIMESTAMP].dt.weekday
        df[constants.F_TS_HOUR] = df[constants.COL_TIMESTAMP].dt.hour
        df[constants.F_BOOK_AGE_AT_TS] = df[constants.F_TS_YEAR] - df[constants.COL_PUBLICATION_YEAR]
    return df


def add_interaction_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Adds engagement counts using the full raw train (including has_read=0)."""
    print("Adding user/book interaction counts (incl. wishlist)...")
    raw_train = pd.read_csv(
        config.RAW_DATA_DIR / constants.TRAIN_FILENAME,
        usecols=[constants.COL_USER_ID, constants.COL_BOOK_ID, constants.COL_HAS_READ],
    )

    user_read = raw_train[raw_train[constants.COL_HAS_READ] == 1].groupby(constants.COL_USER_ID).size()
    user_wishlist = raw_train[raw_train[constants.COL_HAS_READ] == 0].groupby(constants.COL_USER_ID).size()

    user_counts = (
        pd.concat(
            [
                user_read.rename(constants.F_USER_READ_COUNT_ALL),
                user_wishlist.rename(constants.F_USER_WISHLIST_COUNT),
            ],
            axis=1,
        )
        .fillna(0)
        .reset_index()
    )
    user_counts[constants.F_USER_TOTAL_INTERACTIONS] = (
        user_counts[constants.F_USER_READ_COUNT_ALL] + user_counts[constants.F_USER_WISHLIST_COUNT]
    )
    user_counts[constants.F_USER_READ_RATIO] = (
        user_counts[constants.F_USER_READ_COUNT_ALL] / user_counts[constants.F_USER_TOTAL_INTERACTIONS].replace(0, np.nan)
    )

    book_wishlist = (
        raw_train[raw_train[constants.COL_HAS_READ] == 0]
        .groupby(constants.COL_BOOK_ID)
        .size()
        .rename(constants.F_BOOK_WISHLIST_COUNT)
        .reset_index()
    )

    df = df.merge(user_counts, on=constants.COL_USER_ID, how="left")
    df = df.merge(book_wishlist, on=constants.COL_BOOK_ID, how="left")
    return df


def add_aggregate_features(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds user, book, and author aggregate features.

    Uses the training data to compute mean ratings and interaction counts
    to prevent data leakage from the test set.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        train_df (pd.DataFrame): The training portion of the data for calculations.

    Returns:
        pd.DataFrame: The DataFrame with new aggregate features.
    """
    print("Adding aggregate features...")

    global_mean = train_df[config.TARGET].mean()

    # User-based aggregates
    user_agg = train_df.groupby(constants.COL_USER_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    user_agg.columns = [
        constants.COL_USER_ID,
        constants.F_USER_MEAN_RATING,
        constants.F_USER_RATINGS_COUNT,
    ]
    user_agg[constants.F_USER_BIAS] = user_agg[constants.F_USER_MEAN_RATING] - global_mean

    # Book-based aggregates
    book_agg = train_df.groupby(constants.COL_BOOK_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    book_agg.columns = [
        constants.COL_BOOK_ID,
        constants.F_BOOK_MEAN_RATING,
        constants.F_BOOK_RATINGS_COUNT,
    ]
    book_agg[constants.F_BOOK_BIAS] = book_agg[constants.F_BOOK_MEAN_RATING] - global_mean

    # Author-based aggregates
    author_agg = train_df.groupby(constants.COL_AUTHOR_ID)[config.TARGET].agg(["mean", "count"]).reset_index()
    author_agg.columns = [
        constants.COL_AUTHOR_ID,
        constants.F_AUTHOR_MEAN_RATING,
        constants.F_AUTHOR_RATINGS_COUNT,
    ]
    author_agg[constants.F_AUTHOR_BIAS] = author_agg[constants.F_AUTHOR_MEAN_RATING] - global_mean

    # Recency-aware features
    ref_time = train_df[constants.COL_TIMESTAMP].max()

    # Last rating per user and recency/activity
    user_time_stats = (
        train_df.sort_values(constants.COL_TIMESTAMP)
        .groupby(constants.COL_USER_ID)
        .agg(
            last_rating=(config.TARGET, "last"),
            last_ts=(constants.COL_TIMESTAMP, "last"),
            first_ts=(constants.COL_TIMESTAMP, "first"),
        )
        .reset_index()
    )
    user_time_stats[constants.F_USER_RECENCY_DAYS] = (
        (ref_time - user_time_stats["last_ts"]).dt.total_seconds() / 86400.0
    )
    user_time_stats[constants.F_USER_ACTIVITY_DAYS] = (
        (user_time_stats["last_ts"] - user_time_stats["first_ts"]).dt.total_seconds() / 86400.0
    )
    user_time_stats = user_time_stats.rename(columns={"last_rating": constants.F_USER_LAST_RATING})
    user_time_stats = user_time_stats.drop(columns=["last_ts", "first_ts"])

    # Last rating per book and recency
    book_time_stats = (
        train_df.sort_values(constants.COL_TIMESTAMP)
        .groupby(constants.COL_BOOK_ID)
        .agg(
            last_rating=(config.TARGET, "last"),
            last_ts=(constants.COL_TIMESTAMP, "last"),
        )
        .reset_index()
    )
    book_time_stats[constants.F_BOOK_RECENCY_DAYS] = (
        (ref_time - book_time_stats["last_ts"]).dt.total_seconds() / 86400.0
    )
    book_time_stats = book_time_stats.rename(columns={"last_rating": constants.F_BOOK_LAST_RATING})
    book_time_stats = book_time_stats.drop(columns=["last_ts"])

    # Additional stability stats
    user_dispersion = (
        train_df.groupby(constants.COL_USER_ID)[config.TARGET]
        .agg(["std", "median"])
        .reset_index()
        .rename(
            columns={
                "std": constants.F_USER_RATINGS_STD,
                "median": constants.F_USER_RATINGS_MEDIAN,
            }
        )
    )
    book_dispersion = (
        train_df.groupby(constants.COL_BOOK_ID)[config.TARGET]
        .agg(["std", "median"])
        .reset_index()
        .rename(
            columns={
                "std": constants.F_BOOK_RATINGS_STD,
                "median": constants.F_BOOK_RATINGS_MEDIAN,
            }
        )
    )

    # Language and publisher averages help with sparse books/authors
    language_agg = (
        train_df.groupby(constants.COL_LANGUAGE)[config.TARGET]
        .mean()
        .reset_index()
        .rename(columns={config.TARGET: constants.F_LANGUAGE_MEAN_RATING})
    )
    publisher_agg = (
        train_df.groupby(constants.COL_PUBLISHER)[config.TARGET]
        .mean()
        .reset_index()
        .rename(columns={config.TARGET: constants.F_PUBLISHER_MEAN_RATING})
    )

    age_bucket_agg = pd.DataFrame()
    if constants.COL_AGE_BUCKET in train_df.columns:
        age_bucket_agg = (
            train_df.groupby(constants.COL_AGE_BUCKET)[config.TARGET]
            .mean()
            .reset_index()
            .rename(columns={config.TARGET: constants.F_AGE_BUCKET_MEAN_RATING})
        )

    publication_decade_agg = pd.DataFrame()
    if constants.COL_PUBLICATION_YEAR in train_df.columns:
        decade_col = (train_df[constants.COL_PUBLICATION_YEAR] // 10 * 10).astype("float32")
        publication_decade_agg = (
            train_df.assign(_pub_decade=decade_col)
            .groupby("_pub_decade")[config.TARGET]
            .mean()
            .reset_index()
            .rename(columns={"_pub_decade": constants.COL_PUBLICATION_YEAR, config.TARGET: constants.F_PUBLICATION_DECADE_MEAN_RATING})
        )

    month_mean_rating = pd.DataFrame()
    if constants.COL_TIMESTAMP in train_df.columns:
        ts_train = train_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(ts_train[constants.COL_TIMESTAMP]):
            ts_train[constants.COL_TIMESTAMP] = pd.to_datetime(ts_train[constants.COL_TIMESTAMP], errors="coerce")
        ts_train["_ts_month_period"] = ts_train[constants.COL_TIMESTAMP].dt.to_period("M").astype(str)
        month_mean_rating = (
            ts_train.groupby("_ts_month_period")[config.TARGET]
            .mean()
            .reset_index()
            .rename(columns={config.TARGET: constants.F_TS_MONTH_MEAN_RATING})
        )

    genre_agg = pd.DataFrame()
    user_genre_agg = pd.DataFrame()
    if constants.COL_MAIN_GENRE in train_df.columns:
        genre_agg = (
            train_df.groupby(constants.COL_MAIN_GENRE)[config.TARGET]
            .agg(["mean", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": constants.F_GENRE_MEAN_RATING,
                    "count": constants.F_GENRE_RATINGS_COUNT,
                }
            )
        )
        genre_agg[constants.F_GENRE_BIAS] = genre_agg[constants.F_GENRE_MEAN_RATING] - global_mean
        genre_agg.loc[genre_agg[constants.F_GENRE_RATINGS_COUNT] < 3, constants.F_GENRE_MEAN_RATING] = np.nan

        user_genre_agg = (
            train_df.groupby([constants.COL_USER_ID, constants.COL_MAIN_GENRE])[config.TARGET]
            .agg(["mean", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": constants.F_USER_GENRE_MEAN_RATING,
                    "count": constants.F_USER_GENRE_RATINGS_COUNT,
                }
            )
        )
        user_genre_agg.loc[
            user_genre_agg[constants.F_USER_GENRE_RATINGS_COUNT] < 2, constants.F_USER_GENRE_MEAN_RATING
        ] = np.nan

    user_author_agg = (
        train_df.groupby([constants.COL_USER_ID, constants.COL_AUTHOR_ID])[config.TARGET]
        .agg(["mean", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": constants.F_USER_AUTHOR_MEAN_RATING,
                "count": constants.F_USER_AUTHOR_RATINGS_COUNT,
            }
        )
    )
    user_author_agg.loc[
        user_author_agg[constants.F_USER_AUTHOR_RATINGS_COUNT] < 2, constants.F_USER_AUTHOR_MEAN_RATING
    ] = np.nan

    # Merge aggregates into the main dataframe
    df = df.merge(user_agg, on=constants.COL_USER_ID, how="left")
    df = df.merge(book_agg, on=constants.COL_BOOK_ID, how="left")
    df = df.merge(author_agg, on=constants.COL_AUTHOR_ID, how="left")
    df = df.merge(user_time_stats, on=constants.COL_USER_ID, how="left")
    df = df.merge(book_time_stats, on=constants.COL_BOOK_ID, how="left")
    df = df.merge(user_dispersion, on=constants.COL_USER_ID, how="left")
    df = df.merge(book_dispersion, on=constants.COL_BOOK_ID, how="left")
    df = df.merge(language_agg, on=constants.COL_LANGUAGE, how="left")
    df = df.merge(publisher_agg, on=constants.COL_PUBLISHER, how="left")
    if not age_bucket_agg.empty:
        df = df.merge(age_bucket_agg, on=constants.COL_AGE_BUCKET, how="left")
    if not publication_decade_agg.empty:
        df = df.merge(publication_decade_agg, on=constants.COL_PUBLICATION_YEAR, how="left")
    if not month_mean_rating.empty:
        df_ts = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_ts[constants.COL_TIMESTAMP]):
            df_ts[constants.COL_TIMESTAMP] = pd.to_datetime(df_ts[constants.COL_TIMESTAMP], errors="coerce")
        df_ts["_ts_month_period"] = df_ts[constants.COL_TIMESTAMP].dt.to_period("M").astype(str)
        df = df_ts.merge(month_mean_rating, on="_ts_month_period", how="left").drop(columns=["_ts_month_period"])

    if not genre_agg.empty:
        df = df.merge(genre_agg, on=constants.COL_MAIN_GENRE, how="left")
    if not user_genre_agg.empty:
        df = df.merge(user_genre_agg, on=[constants.COL_USER_ID, constants.COL_MAIN_GENRE], how="left")

    df = df.merge(user_author_agg, on=[constants.COL_USER_ID, constants.COL_AUTHOR_ID], how="left")
    return df


def add_genre_features(df: pd.DataFrame, book_genres_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates and adds the count of genres for each book.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        book_genres_df (pd.DataFrame): DataFrame mapping books to genres.

    Returns:
        pd.DataFrame: The DataFrame with the new 'book_genres_count' column.
    """
    print("Adding genre features...")
    book_genres_df = book_genres_df.drop_duplicates(subset=[constants.COL_BOOK_ID, constants.COL_GENRE_ID])

    genre_counts = book_genres_df.groupby(constants.COL_BOOK_ID)[constants.COL_GENRE_ID].count().reset_index()
    genre_counts.columns = [constants.COL_BOOK_ID, constants.F_BOOK_GENRES_COUNT]

    # Deterministic primary genre per book (mode; ties resolved by sorted order)
    primary_genres = (
        book_genres_df.groupby(constants.COL_BOOK_ID)[constants.COL_GENRE_ID]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        .reset_index()
        .rename(columns={constants.COL_GENRE_ID: constants.COL_MAIN_GENRE})
    )

    df = df.merge(genre_counts, on=constants.COL_BOOK_ID, how="left")
    df = df.merge(primary_genres, on=constants.COL_BOOK_ID, how="left")
    return df


def add_text_features(df: pd.DataFrame, train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """Adds TF-IDF features from book descriptions.

    Trains a TF-IDF vectorizer only on training data descriptions to avoid
    data leakage. Applies the vectorizer to all books and merges the features.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        train_df (pd.DataFrame): The training portion for fitting the vectorizer.
        descriptions_df (pd.DataFrame): DataFrame with book descriptions.

    Returns:
        pd.DataFrame: The DataFrame with TF-IDF features added.
    """
    if not config.USE_TFIDF:
        print("Skipping TF-IDF features (USE_TFIDF=0)")
        return df

    print("Adding text features (TF-IDF)...")

    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    vectorizer_path = config.MODEL_DIR / constants.TFIDF_VECTORIZER_FILENAME

    # Get unique books from train set
    train_books = train_df[constants.COL_BOOK_ID].unique()

    # Extract descriptions for training books only
    train_descriptions = descriptions_df[descriptions_df[constants.COL_BOOK_ID].isin(train_books)].copy()
    train_descriptions[constants.COL_DESCRIPTION] = train_descriptions[constants.COL_DESCRIPTION].fillna("")

    # Check if vectorizer already exists (for prediction)
    if vectorizer_path.exists():
        print(f"Loading existing vectorizer from {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
    else:
        # Fit vectorizer on training descriptions only
        print("Fitting TF-IDF vectorizer on training descriptions...")
        vectorizer = TfidfVectorizer(
            max_features=config.TFIDF_MAX_FEATURES,
            min_df=config.TFIDF_MIN_DF,
            max_df=config.TFIDF_MAX_DF,
            ngram_range=config.TFIDF_NGRAM_RANGE,
        )
        vectorizer.fit(train_descriptions[constants.COL_DESCRIPTION])
        # Save vectorizer for use in prediction
        joblib.dump(vectorizer, vectorizer_path)
        print(f"Vectorizer saved to {vectorizer_path}")

    # Transform all book descriptions
    all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]].copy()
    all_descriptions[constants.COL_DESCRIPTION] = all_descriptions[constants.COL_DESCRIPTION].fillna("")

    # Get descriptions in the same order as df[book_id]
    # Create a mapping book_id -> description
    description_map = dict(
        zip(all_descriptions[constants.COL_BOOK_ID], all_descriptions[constants.COL_DESCRIPTION], strict=False)
    )

    # Get descriptions for books in df (in the same order)
    df_descriptions = df[constants.COL_BOOK_ID].map(description_map).fillna("")

    # Transform to TF-IDF features
    tfidf_matrix = vectorizer.transform(df_descriptions)

    # Convert sparse matrix to DataFrame
    tfidf_feature_names = [f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])]
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=tfidf_feature_names,
        index=df.index,
    )

    # Concatenate TF-IDF features with main DataFrame
    df_with_tfidf = pd.concat([df.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)

    print(f"Added {len(tfidf_feature_names)} TF-IDF features.")
    return df_with_tfidf


def add_bert_features(df: pd.DataFrame, _train_df: pd.DataFrame, descriptions_df: pd.DataFrame) -> pd.DataFrame:
    """Adds BERT embeddings from book descriptions.

    Extracts 768-dimensional embeddings using a pre-trained Russian BERT model.
    Embeddings are cached on disk to avoid recomputation on subsequent runs.

    Args:
        df (pd.DataFrame): The main DataFrame to add features to.
        _train_df (pd.DataFrame): The training portion (for consistency, not used for BERT).
        descriptions_df (pd.DataFrame): DataFrame with book descriptions.

    Returns:
        pd.DataFrame: The DataFrame with BERT embeddings added.
    """
    if not config.USE_BERT:
        print("Skipping BERT embeddings (USE_BERT=0)")
        return df

    print("Adding text features (BERT embeddings)...")

    # Ensure model directory exists
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    embeddings_path = config.MODEL_DIR / constants.BERT_EMBEDDINGS_FILENAME

    # Check if embeddings are already cached
    if embeddings_path.exists():
        print(f"Loading cached BERT embeddings from {embeddings_path}")
        embeddings_dict = joblib.load(embeddings_path)
    else:
        print("Computing BERT embeddings (this may take a while)...")
        print(f"Using device: {config.BERT_DEVICE} | batch_size={config.BERT_BATCH_SIZE} | threads={config.TORCH_NUM_THREADS}")

        if torch is not None:
            torch.set_num_threads(config.TORCH_NUM_THREADS)

        # Limit GPU memory usage to prevent OOM errors
        if config.BERT_DEVICE == "cuda" and torch is not None:
            torch.cuda.set_per_process_memory_fraction(config.BERT_GPU_MEMORY_FRACTION)
            print(f"GPU memory limited to {config.BERT_GPU_MEMORY_FRACTION * 100:.0f}% of available memory")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)
        model = AutoModel.from_pretrained(config.BERT_MODEL_NAME)
        model.to(config.BERT_DEVICE)
        model.eval()

        # Prepare descriptions: get unique book_id -> description mapping
        all_descriptions = descriptions_df[[constants.COL_BOOK_ID, constants.COL_DESCRIPTION]].copy()
        all_descriptions[constants.COL_DESCRIPTION] = all_descriptions[constants.COL_DESCRIPTION].fillna("")

        # Get unique books and their descriptions
        unique_books = all_descriptions.drop_duplicates(subset=[constants.COL_BOOK_ID])
        book_ids = unique_books[constants.COL_BOOK_ID].to_numpy()
        descriptions = unique_books[constants.COL_DESCRIPTION].to_numpy().tolist()

        # Initialize embeddings dictionary
        embeddings_dict = {}

        # Process descriptions in batches
        num_batches = (len(descriptions) + config.BERT_BATCH_SIZE - 1) // config.BERT_BATCH_SIZE

        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Processing BERT batches", unit="batch"):
                start_idx = batch_idx * config.BERT_BATCH_SIZE
                end_idx = min(start_idx + config.BERT_BATCH_SIZE, len(descriptions))
                batch_descriptions = descriptions[start_idx:end_idx]
                batch_book_ids = book_ids[start_idx:end_idx]

                # Tokenize batch
                encoded = tokenizer(
                    batch_descriptions,
                    padding=True,
                    truncation=True,
                    max_length=config.BERT_MAX_LENGTH,
                    return_tensors="pt",
                )

                # Move to device
                encoded = {k: v.to(config.BERT_DEVICE) for k, v in encoded.items()}

                # Get model outputs
                outputs = model(**encoded)

                # Mean pooling: average over sequence length dimension
                # outputs.last_hidden_state shape: (batch_size, seq_len, hidden_size)
                attention_mask = encoded["attention_mask"]
                # Expand attention mask to match hidden_size dimension for broadcasting
                attention_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()

                # Sum embeddings, weighted by attention mask
                sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask_expanded, dim=1)
                # Sum attention mask values for normalization
                sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)

                # Mean pooling
                mean_pooled = sum_embeddings / sum_mask

                # Convert to numpy and store
                batch_embeddings = mean_pooled.cpu().numpy()

                for book_id, embedding in zip(batch_book_ids, batch_embeddings, strict=False):
                    embeddings_dict[book_id] = embedding

                # Small pause between batches to let GPU cool down and prevent overheating
                if config.BERT_DEVICE == "cuda":
                    time.sleep(0.2)  # 200ms pause between batches

        # Save embeddings for future use
        joblib.dump(embeddings_dict, embeddings_path)
        print(f"Saved BERT embeddings to {embeddings_path}")

    # Map embeddings to DataFrame rows by book_id
    df_book_ids = df[constants.COL_BOOK_ID].to_numpy()

    # Create embedding matrix
    embeddings_list = []
    for book_id in df_book_ids:
        if book_id in embeddings_dict:
            embeddings_list.append(embeddings_dict[book_id])
        else:
            # Zero embedding for books without descriptions
            embeddings_list.append(np.zeros(config.BERT_EMBEDDING_DIM))

    embeddings_array = np.array(embeddings_list)

    # Optionally compress embeddings with SVD for robustness
    svd_path = config.MODEL_DIR / constants.BERT_SVD_FILENAME
    if svd_path.exists():
        svd = joblib.load(svd_path)
    else:
        svd = TruncatedSVD(
            n_components=min(config.BERT_SVD_COMPONENTS, config.BERT_EMBEDDING_DIM - 1),
            random_state=config.RANDOM_STATE,
        )
        svd.fit(embeddings_array)
        joblib.dump(svd, svd_path)

    bert_svd = svd.transform(embeddings_array)
    bert_feature_names = [f"bert_svd_{i}" for i in range(bert_svd.shape[1])]
    bert_df = pd.DataFrame(bert_svd, columns=bert_feature_names, index=df.index)

    df_with_bert = pd.concat([df.reset_index(drop=True), bert_df.reset_index(drop=True)], axis=1)

    print(f"Added {len(bert_feature_names)} compressed BERT features (SVD).")
    return df_with_bert


def handle_missing_values(df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:  # noqa: C901
    """Fills missing values using a defined strategy.

    Fills missing values for age, aggregated features, and categorical features
    to prepare the DataFrame for model training. Uses metrics from the training
    set (e.g., global mean) to fill NaNs.

    Args:
        df (pd.DataFrame): The DataFrame with missing values.
        train_df (pd.DataFrame): The training data, used for calculating fill metrics.

    Returns:
        pd.DataFrame: The DataFrame with missing values handled.
    """
    print("Handling missing values...")

    # Calculate global mean from training data for filling
    global_mean = train_df[config.TARGET].mean()

    # Fill age with the median
    age_median = df[constants.COL_AGE].median()
    df[constants.COL_AGE] = df[constants.COL_AGE].fillna(age_median)

    # Fill derived numeric helper features
    if constants.COL_BOOK_AGE in df.columns:
        df[constants.COL_BOOK_AGE] = df[constants.COL_BOOK_AGE].replace([np.inf, -np.inf], np.nan)
        df[constants.COL_BOOK_AGE] = df[constants.COL_BOOK_AGE].fillna(df[constants.COL_BOOK_AGE].median())
    if constants.F_BOOK_AGE_AT_TS in df.columns:
        df[constants.F_BOOK_AGE_AT_TS] = df[constants.F_BOOK_AGE_AT_TS].replace([np.inf, -np.inf], np.nan)
        df[constants.F_BOOK_AGE_AT_TS] = df[constants.F_BOOK_AGE_AT_TS].fillna(df[constants.F_BOOK_AGE_AT_TS].median())

    # Fill aggregate features for "cold start" users/items (only if they exist)
    if constants.F_USER_MEAN_RATING in df.columns:
        df[constants.F_USER_MEAN_RATING] = df[constants.F_USER_MEAN_RATING].fillna(global_mean)
    if constants.F_BOOK_MEAN_RATING in df.columns:
        df[constants.F_BOOK_MEAN_RATING] = df[constants.F_BOOK_MEAN_RATING].fillna(global_mean)
    if constants.F_AUTHOR_MEAN_RATING in df.columns:
        df[constants.F_AUTHOR_MEAN_RATING] = df[constants.F_AUTHOR_MEAN_RATING].fillna(global_mean)
    if constants.F_LANGUAGE_MEAN_RATING in df.columns:
        df[constants.F_LANGUAGE_MEAN_RATING] = df[constants.F_LANGUAGE_MEAN_RATING].fillna(global_mean)
    if constants.F_PUBLISHER_MEAN_RATING in df.columns:
        df[constants.F_PUBLISHER_MEAN_RATING] = df[constants.F_PUBLISHER_MEAN_RATING].fillna(global_mean)
    if constants.F_TS_MONTH_MEAN_RATING in df.columns:
        df[constants.F_TS_MONTH_MEAN_RATING] = df[constants.F_TS_MONTH_MEAN_RATING].fillna(global_mean)
    if constants.F_GENRE_MEAN_RATING in df.columns:
        df[constants.F_GENRE_MEAN_RATING] = df[constants.F_GENRE_MEAN_RATING].fillna(global_mean)
    if constants.F_GENRE_BIAS in df.columns:
        df[constants.F_GENRE_BIAS] = df[constants.F_GENRE_BIAS].fillna(0)
    if constants.F_AGE_BUCKET_MEAN_RATING in df.columns:
        df[constants.F_AGE_BUCKET_MEAN_RATING] = df[constants.F_AGE_BUCKET_MEAN_RATING].fillna(global_mean)
    if constants.F_PUBLICATION_DECADE_MEAN_RATING in df.columns:
        df[constants.F_PUBLICATION_DECADE_MEAN_RATING] = df[constants.F_PUBLICATION_DECADE_MEAN_RATING].fillna(global_mean)

    if constants.F_USER_RATINGS_COUNT in df.columns:
        df[constants.F_USER_RATINGS_COUNT] = df[constants.F_USER_RATINGS_COUNT].fillna(0)
    if constants.F_BOOK_RATINGS_COUNT in df.columns:
        df[constants.F_BOOK_RATINGS_COUNT] = df[constants.F_BOOK_RATINGS_COUNT].fillna(0)
    if constants.F_AUTHOR_RATINGS_COUNT in df.columns:
        df[constants.F_AUTHOR_RATINGS_COUNT] = df[constants.F_AUTHOR_RATINGS_COUNT].fillna(0)
    if constants.F_GENRE_RATINGS_COUNT in df.columns:
        df[constants.F_GENRE_RATINGS_COUNT] = df[constants.F_GENRE_RATINGS_COUNT].fillna(0)
    if constants.F_USER_ACTIVITY_DAYS in df.columns:
        df[constants.F_USER_ACTIVITY_DAYS] = df[constants.F_USER_ACTIVITY_DAYS].fillna(0)
    if constants.F_USER_RECENCY_DAYS in df.columns:
        df[constants.F_USER_RECENCY_DAYS] = df[constants.F_USER_RECENCY_DAYS].fillna(df[constants.F_USER_RECENCY_DAYS].median())
    if constants.F_BOOK_RECENCY_DAYS in df.columns:
        df[constants.F_BOOK_RECENCY_DAYS] = df[constants.F_BOOK_RECENCY_DAYS].fillna(df[constants.F_BOOK_RECENCY_DAYS].median())
    if constants.F_USER_LAST_RATING in df.columns:
        df[constants.F_USER_LAST_RATING] = df[constants.F_USER_LAST_RATING].fillna(global_mean)
    if constants.F_BOOK_LAST_RATING in df.columns:
        df[constants.F_BOOK_LAST_RATING] = df[constants.F_BOOK_LAST_RATING].fillna(global_mean)
    if constants.F_USER_BIAS in df.columns:
        df[constants.F_USER_BIAS] = df[constants.F_USER_BIAS].fillna(0)
    if constants.F_BOOK_BIAS in df.columns:
        df[constants.F_BOOK_BIAS] = df[constants.F_BOOK_BIAS].fillna(0)
    if constants.F_AUTHOR_BIAS in df.columns:
        df[constants.F_AUTHOR_BIAS] = df[constants.F_AUTHOR_BIAS].fillna(0)
    if constants.F_USER_TOTAL_INTERACTIONS in df.columns:
        df[constants.F_USER_TOTAL_INTERACTIONS] = df[constants.F_USER_TOTAL_INTERACTIONS].fillna(0)
    if constants.F_USER_WISHLIST_COUNT in df.columns:
        df[constants.F_USER_WISHLIST_COUNT] = df[constants.F_USER_WISHLIST_COUNT].fillna(0)
    if constants.F_USER_READ_COUNT_ALL in df.columns:
        df[constants.F_USER_READ_COUNT_ALL] = df[constants.F_USER_READ_COUNT_ALL].fillna(0)
    if constants.F_USER_READ_RATIO in df.columns:
        df[constants.F_USER_READ_RATIO] = df[constants.F_USER_READ_RATIO].fillna(0)
    if constants.F_BOOK_WISHLIST_COUNT in df.columns:
        df[constants.F_BOOK_WISHLIST_COUNT] = df[constants.F_BOOK_WISHLIST_COUNT].fillna(0)

    if constants.F_USER_RATINGS_STD in df.columns:
        df[constants.F_USER_RATINGS_STD] = df[constants.F_USER_RATINGS_STD].fillna(0)
    if constants.F_BOOK_RATINGS_STD in df.columns:
        df[constants.F_BOOK_RATINGS_STD] = df[constants.F_BOOK_RATINGS_STD].fillna(0)
    if constants.F_USER_RATINGS_MEDIAN in df.columns:
        df[constants.F_USER_RATINGS_MEDIAN] = df[constants.F_USER_RATINGS_MEDIAN].fillna(global_mean)
    if constants.F_BOOK_RATINGS_MEDIAN in df.columns:
        df[constants.F_BOOK_RATINGS_MEDIAN] = df[constants.F_BOOK_RATINGS_MEDIAN].fillna(global_mean)
    if constants.F_USER_GENRE_MEAN_RATING in df.columns:
        df[constants.F_USER_GENRE_MEAN_RATING] = df[constants.F_USER_GENRE_MEAN_RATING].fillna(global_mean)
    if constants.F_USER_GENRE_RATINGS_COUNT in df.columns:
        df[constants.F_USER_GENRE_RATINGS_COUNT] = df[constants.F_USER_GENRE_RATINGS_COUNT].fillna(0)
    if constants.F_USER_AUTHOR_MEAN_RATING in df.columns:
        df[constants.F_USER_AUTHOR_MEAN_RATING] = df[constants.F_USER_AUTHOR_MEAN_RATING].fillna(global_mean)
    if constants.F_USER_AUTHOR_RATINGS_COUNT in df.columns:
        df[constants.F_USER_AUTHOR_RATINGS_COUNT] = df[constants.F_USER_AUTHOR_RATINGS_COUNT].fillna(0)

    # Fill missing avg_rating from book_data with global mean
    df[constants.COL_AVG_RATING] = df[constants.COL_AVG_RATING].fillna(global_mean)

    # Fill genre counts with 0
    df[constants.F_BOOK_GENRES_COUNT] = df[constants.F_BOOK_GENRES_COUNT].fillna(0)

    # Fill TF-IDF features with 0 (for books without descriptions)
    tfidf_cols = [col for col in df.columns if col.startswith("tfidf_")]
    for col in tfidf_cols:
        df[col] = df[col].fillna(0.0)

    # Fill BERT features with 0 (for books without descriptions)
    bert_cols = [col for col in df.columns if col.startswith("bert_")]
    for col in bert_cols:
        df[col] = df[col].fillna(0.0)
    bert_svd_cols = [col for col in df.columns if col.startswith("bert_svd_")]
    for col in bert_svd_cols:
        df[col] = df[col].fillna(0.0)
    latent_cols = [col for col in df.columns if col.startswith("lf_user_") or col.startswith("lf_book_")]
    for col in latent_cols:
        df[col] = df[col].fillna(0.0)

    # Fill remaining categorical features with a special value
    for col in config.CAT_FEATURES:
        if col in df.columns:
            if df[col].dtype.name in ("category", "object") and df[col].isna().any():
                df[col] = df[col].astype(str).fillna(constants.MISSING_CAT_VALUE).astype("category")
            elif pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].isna().any():
                df[col] = df[col].fillna(constants.MISSING_NUM_VALUE)

    # Fill timestamp-derived numerics with 0 where missing (test rows)
    for col in [constants.F_TS_YEAR, constants.F_TS_MONTH, constants.F_TS_WEEKDAY, constants.F_TS_HOUR]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def create_features(
    df: pd.DataFrame, book_genres_df: pd.DataFrame, descriptions_df: pd.DataFrame, include_aggregates: bool = False
) -> pd.DataFrame:
    """Runs the full feature engineering pipeline.

    This function orchestrates the calls to add aggregate features (optional), genre
    features, text features (TF-IDF and BERT), and handle missing values.

    Args:
        df (pd.DataFrame): The merged DataFrame from `data_processing`.
        book_genres_df (pd.DataFrame): DataFrame mapping books to genres.
        descriptions_df (pd.DataFrame): DataFrame with book descriptions.
        include_aggregates (bool): If True, compute aggregate features. Defaults to False.
            Aggregates are typically computed separately during training to avoid data leakage.

    Returns:
        pd.DataFrame: The final DataFrame with all features engineered.
    """
    print("Starting feature engineering pipeline...")
    print(f"Text features -> TF-IDF: {config.USE_TFIDF}, BERT: {config.USE_BERT}")
    train_df = df[df[constants.COL_SOURCE] == constants.VAL_SOURCE_TRAIN].copy()

    # Basic deterministic enrichments
    df = add_basic_features(df)
    df = add_interaction_counts(df)

    # Aggregate features are computed separately during training to ensure
    # no data leakage from validation set timestamps
    if include_aggregates:
        df = add_aggregate_features(df, train_df)

    df = add_genre_features(df, book_genres_df)
    df = add_text_features(df, train_df, descriptions_df)
    df = add_bert_features(df, train_df, descriptions_df)
    df = handle_missing_values(df, train_df)

    # Convert categorical columns to pandas 'category' dtype for LightGBM
    for col in config.CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print("Feature engineering complete.")
    return df
