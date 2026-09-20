import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from data_processing import FEATURE_COLUMNS, build_label, clean, engineer_features  # noqa: E402


def _toy_df():
    """Two users, 7 posts each, so we can hand-check the rolling label."""
    rows = []
    for user in [1, 2]:
        for i in range(7):
            rows.append(
                {
                    "User uuid": user,
                    "Likes": 100 + i * 10,
                    "Days passed from post": 100 - i,
                    "Likes Score": 0.5,
                    "Type": "Images" if i % 2 == 0 else "Video",
                    "Numer of Tags": i,
                    "Numer of Comments": i * 2,
                    "Date Posted": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
                    "Year": 2024,
                    "Month": 1,
                    "Day": i + 1,
                    "Hour": 10 + i,
                    "Minute": 0,
                }
            )
    return pd.DataFrame(rows)


def test_clean_fixes_images_typo():
    df = clean(_toy_df())
    assert set(df["Type"].unique()) <= {"Image", "Video"}


def test_engineer_features_has_expected_columns():
    df = engineer_features(clean(_toy_df()))
    for col in FEATURE_COLUMNS:
        assert col in df.columns
    assert df[FEATURE_COLUMNS].isna().sum().sum() == 0


def test_build_label_first_posts_are_unpopular():
    df = engineer_features(clean(_toy_df()))
    df = build_label(df)
    # Each user's first ROLLING_WINDOW posts have no history -> label 0
    first_five_per_user = df.groupby("User uuid").head(5)
    assert (first_five_per_user["popularity"] == 0).all()


def test_build_label_is_binary():
    df = engineer_features(clean(_toy_df()))
    df = build_label(df)
    assert set(df["popularity"].unique()) <= {0, 1}


def test_days_since_prev_post_non_negative():
    df = engineer_features(clean(_toy_df()))
    assert (df["days_since_prev_post"] >= 0).all()
