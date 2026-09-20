"""
predict.py
----------
Load the trained model and predict whether a hypothetical post will be
"popular" (1) or "unpopular" (0), given its planned metadata.

Example:
    python src/predict.py --num_tags 5 --num_comments 3 --media_type image \
        --hour 19 --day_of_week 4 --month 6 --days_since_prev_post 2 \
        --prev_avg_comments_5 6 --prev_avg_tags_5 4 --user_avg_comments_so_far 5.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_model():
    model_path = MODELS_DIR / "best_model.joblib"
    features_path = MODELS_DIR / "feature_columns.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run `python src/train.py` first."
        )
    model = joblib.load(model_path)
    with open(features_path) as f:
        feature_columns = json.load(f)
    return model, feature_columns


def predict_one(model, feature_columns, values: dict) -> dict:
    row = pd.DataFrame([{col: values[col] for col in feature_columns}])
    pred = int(model.predict(row)[0])
    proba = float(model.predict_proba(row)[0, 1]) if hasattr(model, "predict_proba") else None
    return {"popularity": pred, "probability_popular": proba}


def main():
    parser = argparse.ArgumentParser(description="Predict popularity of a hypothetical post.")
    parser.add_argument("--num_tags", type=float, required=True)
    parser.add_argument("--num_comments", type=float, required=True)
    parser.add_argument("--media_type", choices=["image", "video"], required=True)
    parser.add_argument("--hour", type=int, required=True, help="0-23")
    parser.add_argument("--day_of_week", type=int, required=True, help="0=Mon .. 6=Sun")
    parser.add_argument("--month", type=int, required=True, help="1-12")
    parser.add_argument("--days_since_prev_post", type=float, default=1.0)
    parser.add_argument("--prev_avg_comments_5", type=float, default=0.0)
    parser.add_argument("--prev_avg_tags_5", type=float, default=0.0)
    parser.add_argument("--user_avg_comments_so_far", type=float, default=0.0)
    args = parser.parse_args()

    model, feature_columns = load_model()

    values = {
        "num_tags": args.num_tags,
        "num_comments": args.num_comments,
        "media_type": 1 if args.media_type == "video" else 0,
        "hour": args.hour,
        "day_of_week": args.day_of_week,
        "is_weekend": 1 if args.day_of_week in (5, 6) else 0,
        "month": args.month,
        "days_since_prev_post": args.days_since_prev_post,
        "prev_avg_comments_5": args.prev_avg_comments_5,
        "prev_avg_tags_5": args.prev_avg_tags_5,
        "user_avg_comments_so_far": args.user_avg_comments_so_far,
    }

    result = predict_one(model, feature_columns, values)
    label = "POPULAR" if result["popularity"] == 1 else "NOT POPULAR"
    print(f"Prediction: {label}")
    if result["probability_popular"] is not None:
        print(f"Probability of being popular: {result['probability_popular']:.2%}")


if __name__ == "__main__":
    main()
