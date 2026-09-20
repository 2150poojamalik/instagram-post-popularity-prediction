"""
streamlit_app.py
-----------------
Small interactive demo around the trained model.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from predict import load_model, predict_one  # noqa: E402

st.set_page_config(page_title="Instagram Post Popularity Predictor", page_icon="📸")
st.title("📸 Instagram Post Popularity Predictor")
st.caption(
    "Predicts whether a future post will beat the account's recent average "
    "engagement, based on planned posting metadata."
)

try:
    model, feature_columns = load_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

col1, col2 = st.columns(2)
with col1:
    num_tags = st.number_input("Number of tags/hashtags", min_value=0, value=5)
    num_comments = st.number_input("Expected number of comments", min_value=0, value=3)
    media_type = st.selectbox("Media type", ["image", "video"])
    hour = st.slider("Posting hour (0-23)", 0, 23, 19)
with col2:
    day_of_week = st.selectbox(
        "Day of week", options=list(range(7)),
        format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d],
    )
    month = st.slider("Month", 1, 12, 6)
    days_since_prev_post = st.number_input("Days since previous post", min_value=0.0, value=2.0)

st.subheader("Recent account activity (optional, improves accuracy)")
prev_avg_comments_5 = st.number_input("Avg comments on last 5 posts", min_value=0.0, value=5.0)
prev_avg_tags_5 = st.number_input("Avg tags on last 5 posts", min_value=0.0, value=4.0)
user_avg_comments_so_far = st.number_input("All-time avg comments for this account", min_value=0.0, value=5.0)

if st.button("Predict popularity", type="primary"):
    values = {
        "num_tags": num_tags,
        "num_comments": num_comments,
        "media_type": 1 if media_type == "video" else 0,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": 1 if day_of_week in (5, 6) else 0,
        "month": month,
        "days_since_prev_post": days_since_prev_post,
        "prev_avg_comments_5": prev_avg_comments_5,
        "prev_avg_tags_5": prev_avg_tags_5,
        "user_avg_comments_so_far": user_avg_comments_so_far,
    }
    result = predict_one(model, feature_columns, values)
    if result["popularity"] == 1:
        st.success(f"🎉 Likely POPULAR ({result['probability_popular']:.1%} confidence)")
    else:
        st.warning(f"📉 Likely NOT popular ({1 - result['probability_popular']:.1%} confidence)")
