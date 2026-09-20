#  Instagram Post Popularity Prediction

Predicts whether a future Instagram post will out-perform an account's own
recent engagement, using only metadata that's known **before** the post
goes live (tags, timing, media type, posting cadence) — no image or caption
analysis required.

This is a rebuilt, extended version of an earlier prototype of mine (the
original script is kept for reference in [`legacy/`](legacy)). That first
version used a single `RandomForestClassifier` with default settings and a
raw nested Python loop for labelling, reaching **~54% accuracy**. This
version:

| | Original prototype | This repo |
|---|---|---|
| Labelling | 40-line nested `for` loop over `csv.reader` rows | Vectorised `pandas` groupby/rolling (~200x faster) |
| Features | `User uuid`, `Number of Tags`, `Media Type` | + posting hour/day/month, weekend flag, posting cadence, prior engagement trend (leakage-checked) |
| Models compared | 1 (Random Forest, default params) | 4 (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting) |
| Evaluation | Accuracy only, single split | Accuracy, Precision, Recall, F1, ROC-AUC |
| Best result | ~54.5% accuracy | **~74% accuracy / 0.69 F1 / 0.82 ROC-AUC** |
| Usability | Colab-only, hardcoded Google Drive paths | CLI training/prediction scripts + Streamlit demo, runs anywhere |

## Results

Trained and evaluated on the full 178,922-post dataset (1,089 accounts), 80/20 stratified split:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.593 | 0.751 | 0.090 | 0.160 | 0.640 |
| Decision Tree | 0.726 | 0.691 | 0.662 | 0.676 | 0.788 |
| Random Forest | 0.732 | 0.735 | 0.596 | 0.658 | 0.804 |
| **Gradient Boosting** | **0.742** | 0.726 | 0.648 | **0.685** | **0.816** |

Gradient Boosting and Random Forest are close; Gradient Boosting wins on F1
and ROC-AUC and produces a far smaller model file, so it's saved as the
default in `models/best_model.joblib`. (Random Forest's tree depth/leaf-size
are deliberately capped — an unconstrained forest scores marginally higher
but balloons to 1GB+, which won't fit in a git repo and mostly just
memorizes the training data.) Re-run `python src/train.py` any time to
regenerate this table (see `models/metrics.json`).

## Why the numbers changed

The label itself (`popularity` = 1 if a post beat the rolling average of the
account's previous 5 posts) is unchanged from the original prototype — that
logic is preserved intentionally so results are comparable. What changed is:

1. **More informative, leakage-safe features.** The original only used
   account ID, tag count and media type. This version adds posting-time
   features (hour, day of week, weekend, month) and prior-engagement trend
   features (rolling comment average, tagging habits, posting cadence) —
   all grounded in well-known drivers of social media engagement (hashtags,
   timing, media type, posting consistency).
2. **No target leakage.** `Likes` and the rolling-average-of-likes used to
   build the label are excluded from the feature set.
3. **Model comparison instead of a single default model.** Gradient
   Boosting wins here, but Random Forest is close behind and Logistic
   Regression is included as a fast, interpretable baseline.
4. **Proper evaluation.** Accuracy alone is misleading on a dataset that's
   57/43 balanced — F1 and ROC-AUC are reported alongside it.

## Project structure

```
instagram-post-popularity-prediction/
├── data/
│   └── Instagram_Data.csv        # raw dataset (1,089 users, 178,922 posts)
├── src/
│   ├── data_processing.py        # cleaning, feature engineering, labelling
│   ├── train.py                  # trains + compares models, saves the best one
│   └── predict.py                # CLI: predict popularity for one hypothetical post
├── app/
│   └── streamlit_app.py          # interactive web demo
├── models/
│   ├── best_model.joblib         # trained model (ships pre-trained, ready to use)
│   ├── metrics.json              # evaluation metrics for every model tried
│   └── feature_columns.json
├── tests/
│   └── test_data_processing.py   # unit tests for the feature/label pipeline
├── docs/
│   └── REPORT.md                 # concise write-up: problem, approach, results
├── legacy/
│   └── original_colab_notebook.py  # my original, unmodified prototype
├── .github/workflows/ci.yml      # lint + test on every push
├── requirements.txt
└── README.md
```

## Getting started

```bash
git clone https://github.com/<your-username>/instagram-post-popularity-prediction.git
cd instagram-post-popularity-prediction
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Use the pre-trained model right away

```bash
python src/predict.py \
  --num_tags 5 --num_comments 3 --media_type image \
  --hour 19 --day_of_week 4 --month 6 \
  --days_since_prev_post 2 \
  --prev_avg_comments_5 6 --prev_avg_tags_5 4 --user_avg_comments_so_far 5.5
```

### Or launch the interactive demo

```bash
streamlit run app/streamlit_app.py
```

### Retrain from scratch

```bash
python src/train.py --data data/Instagram_Data.csv          # full run, all models
python src/train.py --data data/Instagram_Data.csv --sample 20000 --skip-svm --skip-cv  # quick smoke test
python src/train.py --data data/Instagram_Data.csv --tune    # + hyperparameter search on Random Forest
```

### Run the tests

```bash
pytest tests/ -v
```

## Dataset

1,089 Instagram accounts, 178,922 posts, columns: user ID, likes, days since
posted, media type (image/video), tag count, comment count, and a full
timestamp. See [`docs/REPORT.md`](docs/REPORT.md) for the full write-up of
the problem, approach, and results.

## Roadmap / possible extensions

- [ ] Add caption text features (sentiment, length, emoji count) if caption
      data becomes available
- [ ] Try LightGBM / XGBoost for a speed and accuracy comparison
- [ ] Add SHAP-based feature importance plots
- [ ] Per-account fine-tuning for high-volume accounts
- [ ] Package `predict.py` behind a small REST API (FastAPI) for integration
      into other tools

## License

MIT — see [LICENSE](LICENSE).
