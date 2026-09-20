# Project Report: Instagram Post Popularity Prediction

## Overview

This project predicts whether a future Instagram post will outperform an
account's own recent engagement — before the post even goes live — using
only metadata that's known in advance: tag count, media type, posting time,
and recent posting activity. No image or caption analysis involved.

## Problem statement

Social media performance is inconsistent and hard to plan around. Rather
than trying to predict an exact like-count (which depends on too many
external factors — follower mood, algorithm changes, trends of the day —
to model reliably from metadata alone), this project treats it as a
**binary classification problem**: will this post beat the account's
recent average engagement, or not?

A post is labelled **popular (1)** if its like-count exceeds the moving
average of the account's previous 5 posts, and **unpopular (0)** otherwise.

## Dataset

- 1,089 Instagram accounts, 178,922 posts
- Fields: likes, comment count, tag count, media type (image/video), and a
  full posting timestamp
- No missing values; the only cleanup needed was a typo fix (`"Images"` →
  `"Image"`) in the media type column

## Approach

1. **Feature engineering** — instead of only account ID, tag count, and
   media type, I added:
   - posting hour, day of week, weekend flag, month
   - days since the account's previous post (posting cadence)
   - rolling average of comments/tags on the account's last 5 posts
   - the account's all-time average comment count so far

   All of these are computable *before* a post is published, so none of
   them leak the label.

2. **Labelling** — rebuilt as a vectorised pandas `groupby().rolling()`
   operation instead of a manual nested loop. Same rule, much faster and
   easier to verify with unit tests.

3. **Modeling** — trained and compared four classifiers (Logistic
   Regression, Decision Tree, Random Forest, Gradient Boosting) with an
   80/20 stratified split, and evaluated each on accuracy, precision,
   recall, F1, and ROC-AUC — not accuracy alone, since the classes are only
   roughly balanced (57/43).

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.593 | 0.751 | 0.090 | 0.160 | 0.640 |
| Decision Tree | 0.726 | 0.691 | 0.662 | 0.676 | 0.788 |
| Random Forest | 0.732 | 0.735 | 0.596 | 0.658 | 0.804 |
| **Gradient Boosting** | **0.742** | 0.726 | 0.648 | **0.685** | **0.816** |

Gradient Boosting is the model shipped in `models/best_model.joblib`.

## What I'd improve next

- Add caption-based features (sentiment, length, emoji count) if caption
  text becomes available
- Compare against LightGBM/XGBoost
- Add SHAP-based feature importance for interpretability
- Wrap `predict.py` in a small REST API for easier integration elsewhere

See the main [README](../README.md) for setup and usage instructions.
