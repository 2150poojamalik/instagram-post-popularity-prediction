"""
train.py
--------
Trains and compares several classifiers on the Instagram popularity dataset,
picks the best one by cross-validated F1 score, and saves it to disk.

The original project only ever tried a single RandomForestClassifier with
default hyperparameters and got ~54% accuracy on a near-balanced binary
target (i.e. barely above a coin flip). This script:

  * compares Logistic Regression, Decision Tree, Random Forest, Gradient
    Boosting and (optionally) SVM
  * uses stratified train/test split + 5-fold cross-validation instead of
    a single random split
  * reports accuracy, precision, recall, F1 and ROC-AUC (not just accuracy)
  * scales features for the models that need it (Logistic Regression, SVM)
  * runs a small hyperparameter search on the best-performing model family
  * saves the trained model + feature list + metrics to models/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_processing import FEATURE_COLUMNS, build_dataset

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def get_candidate_models(random_state: int = 42) -> dict:
    return {
        "logistic_regression": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
        ),
        "decision_tree": DecisionTreeClassifier(max_depth=8, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "svm_rbf": Pipeline(
            [("scaler", StandardScaler()), ("clf", SVC(probability=True, kernel="rbf"))]
        ),
    }


def evaluate_model(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Instagram popularity models.")
    parser.add_argument("--data", default="data/Instagram_Data.csv", help="Path to raw CSV.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--sample", type=int, default=None, help="Optional row subsample for quick runs.")
    parser.add_argument(
        "--skip-svm", action="store_true", help="Skip SVM (slow on large datasets)."
    )
    parser.add_argument(
        "--skip-cv", action="store_true", help="Skip 5-fold cross-validation (faster, single split only)."
    )
    parser.add_argument(
        "--tune", action="store_true", help="Run a small GridSearchCV on the best model family."
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)

    print(f"Loading + engineering features from {args.data} ...")
    dataset = build_dataset(args.data)
    if args.sample:
        dataset = dataset.sample(n=min(args.sample, len(dataset)), random_state=42)
    print(f"Dataset shape: {dataset.shape}")
    print("Label balance:\n", dataset["popularity"].value_counts(normalize=True))

    X = dataset[FEATURE_COLUMNS]
    y = dataset["popularity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    candidates = get_candidate_models()
    if args.skip_svm:
        candidates.pop("svm_rbf", None)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in candidates.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        metrics["train_seconds"] = round(time.time() - t0, 2)
        if not args.skip_cv:
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
            metrics["cv_f1_mean"] = float(cv_scores.mean())
            metrics["cv_f1_std"] = float(cv_scores.std())
        results[name] = metrics
        print(f"\n[{name}]  ({metrics['train_seconds']}s)")
        for k, v in metrics.items():
            if k != "train_seconds":
                print(f"  {k:10s}: {v:.4f}")

    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = candidates[best_name]
    print(f"\nBest model by F1 score: {best_name} (F1={results[best_name]['f1']:.4f})")

    if args.tune and best_name == "random_forest":
        print("\nRunning GridSearchCV on random_forest ...")
        param_grid = {
            "n_estimators": [200, 400],
            "max_depth": [None, 10, 20],
            "min_samples_leaf": [1, 2, 4],
        }
        search = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid,
            scoring="f1",
            cv=3,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        results["random_forest_tuned"] = evaluate_model(best_model, X_test, y_test)
        print("Best params:", search.best_params_)
        print("Tuned metrics:", results["random_forest_tuned"])

    # Persist model + metadata
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    print(f"\nSaved best model ({best_name}) to {MODELS_DIR / 'best_model.joblib'}")
    print(f"Saved metrics to {MODELS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
