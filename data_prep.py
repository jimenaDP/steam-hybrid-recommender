"""
General data preprocessing functions
"""
#Imports
import ast
import numpy as np
import pandas as pd


def load_raw_data(dataset_dir="dataset"):
    """Load the raw data"""
    extended_games = pd.read_csv(f"{dataset_dir}/extended_games.csv")
    train_data = pd.read_csv(f"{dataset_dir}/train_interactions.csv")
    test_data = pd.read_csv(f"{dataset_dir}/test_interactions_in.csv")
    return extended_games, train_data, test_data


def parse_publisher_list(x):
    """Parse the string-list publishers field into a real list"""
    if not isinstance(x, str) or x.strip() == "":
        return []
    try:
        parsed = ast.literal_eval(x)
        return parsed if isinstance(parsed, list) else [str(parsed)]
    except (ValueError, SyntaxError):
        return [x.strip()]


def preprocess_extended_games(extended_games_raw, path="dataset/extended_games_clean.csv"):
    """Select relevant columns, fill missing text and parse publishers."""
    reduced_cols = [
        "item_id", "item_name", "publishers", "positive", "negative",
        "recommendations", "short_description", "about_the_game",
        "detailed_description", "reviews"
    ]
    extended_games = extended_games_raw[reduced_cols].copy()

    #safety cleaning
    n_missing_id = extended_games["item_id"].isna().sum()
    n_missing_name = extended_games["item_name"].isna().sum()
    n_dupes_id = extended_games["item_id"].duplicated().sum()
    print(f"BEFORE CLEANING: {n_missing_id} missing item_id, {n_missing_name} missing item_name, "
          f"{n_dupes_id} duplicate item_id")

    extended_games = extended_games.dropna(subset=["item_id", "item_name"])
    extended_games = extended_games.drop_duplicates(subset=["item_id"])

    assert extended_games["item_id"].notna().all(), "missing item_id remains after dropna"
    assert extended_games["item_name"].notna().all(), "missing item_name remains after dropna"
    assert not extended_games["item_id"].duplicated().any(), "duplicate item_id remains after drop_duplicates"

    print("AFTER CLEANING: item_id and item_name are unique non-null")

    #missing text fields
    miss_text_cols = extended_games.columns[extended_games.isna().any()]
    print(f"Columns with missing values BEFORE: {list(miss_text_cols)}")
    extended_games[miss_text_cols] = extended_games[miss_text_cols].fillna("")
    assert extended_games.columns[extended_games.isna().any()].empty, \
        "missing values remain after fill"
    print("AFTER: Confirmed no missing values remain")

    #parse column publishers
    extended_games["publishers_list"] = extended_games["publishers"].apply(parse_publisher_list)
    extended_games["publisher"] = extended_games["publishers_list"].apply(
        lambda x: x[0] if x else None
    )
    n_no_publisher = extended_games["publisher"].isna().sum()
    print(f"AFTER: {n_no_publisher} games with no parsed publisher")

    extended_games.to_csv(path, index=False)
    return extended_games


def preprocess_train_data(train_data_raw, path="dataset/train_data_clean.csv"):
    n_dupes = train_data_raw.duplicated(subset=["user_id", "item_id"]).sum() #Checking for dupes
    print(f"BEFORE CLEANING: {n_dupes} duplicate (user_id, item_id) pairs found")

    #keep maximum playtime value
    train_data = train_data_raw.sort_values("playtime", ascending=False) \
                                    .drop_duplicates(subset=["user_id", "item_id"], keep="first") 

    n_dupes_after = train_data.duplicated(subset=["user_id", "item_id"]).sum()
    print(f"AFTER: {n_dupes_after} duplicate pairs remaining after dedup")
    assert n_dupes_after == 0, "duplicates still present after drop_duplicates — this shouldn't happen"


    n_negative = (train_data["playtime"] < 0).sum()
    print(f"AFTER: {n_negative} rows with negative playtime")
    assert n_negative == 0, f"{n_negative} rows have negative playtime — investigate before proceeding"

    #checking for missing values
    cols_with_nan = train_data.columns[train_data.isna().any()]
    print(f"Columns with missing values in training data: {list(cols_with_nan)}")
    assert cols_with_nan.empty, f"missing values found in: {list(cols_with_nan)}"

    train_data.to_csv(path, index=False)
    return train_data


def split_test(
    test_data: pd.DataFrame,
    user_col="user_id",
    min_interactions=2, #users with <2 interactions get dropped
    hold_out_frac=0.2,
    seed=42
)-> tuple:
    """
    Split test_data into fold-in and ground truth.

    - For each user:
        * if interactions < min_interactions → drop user
        * else: hold_out_frac to hold_out (min 1), rest to fold_in
    """

    rng = np.random.default_rng(seed)

    fold_in_rows = []
    hold_out_rows = []

    for u, df_u in test_data.groupby(user_col):
        n = len(df_u)#number of interactions    

        # skip users with too few interactions
        if n < min_interactions:
            continue

        # shuffle user interactions
        idx = rng.permutation(n)

        # number of items in hold_out
        n_out = max(1, int(np.ceil(hold_out_frac * n)))

        out_idx = idx[:n_out]
        in_idx = idx[n_out:]

        hold_out_rows.append(df_u.iloc[out_idx])
        fold_in_rows.append(df_u.iloc[in_idx])

    fold_in = pd.concat(fold_in_rows, ignore_index=True)
    hold_out = pd.concat(hold_out_rows, ignore_index=True)

    return fold_in, hold_out
