import os
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Driver Claim Prediction", layout="wide")

API_URL = os.getenv("API_URL", "http://localhost:8000")


@st.cache_resource
def load_local_model() -> Tuple[Optional[object], List[str]]:
    try:
        model = joblib.load("models/model_pipeline.pkl")
        features = joblib.load("models/selected_features.pkl")
        if model and features:
            return model, features
    except Exception:
        pass

    try:
        from train import train_model

        train_model()
        model = joblib.load("models/model_pipeline.pkl")
        features = joblib.load("models/selected_features.pkl")
        return model, features
    except Exception:
        return None, []


@st.cache_data
def get_features_from_api() -> List[str]:
    try:
        response = requests.get(f"{API_URL}/features", timeout=5)
        if response.ok:
            return response.json()
    except requests.RequestException:
        pass
    return []


def predict_locally(feature_values: Dict[str, float]) -> Tuple[int, float]:
    model, features = load_local_model()
    if not model or not features:
        raise ValueError("Model artifacts are not available.")

    normalized = {}
    for feature in features:
        if feature not in feature_values:
            raise KeyError(f"Missing feature: {feature}")
        normalized[feature] = float(feature_values[feature])

    input_df = pd.DataFrame([normalized], columns=features)
    probability = float(model.predict_proba(input_df)[0, 1])
    prediction = int(model.predict(input_df)[0])
    return prediction, probability


def predict_with_fallback(feature_values: Dict[str, float]) -> Tuple[int, float]:
    try:
        response = requests.post(f"{API_URL}/predict", json={"features": feature_values}, timeout=10)
        if response.ok:
            result = response.json()
            return int(result["prediction"]), float(result["probability"])
    except requests.RequestException:
        pass

    return predict_locally(feature_values)


st.title("🚗 Safe Driver Claim Prediction")
st.caption("Estimate whether a driver is likely to file an insurance claim based on the most predictive input features.")

features = get_features_from_api() or load_local_model()[1]

if not features:
    st.error("No feature list is available. Please ensure the model exists or start the API server.")
    st.stop()

user_input: Dict[str, float] = {}
with st.form("prediction_form"):
    st.subheader("Top model features")
    cols = st.columns(2)

    for index, feature in enumerate(features):
        column = cols[index % 2]
        if feature.endswith("_bin"):
            user_input[feature] = column.selectbox(f"{feature} (Binary)", options=[0, 1], index=0)
        elif feature.endswith("_cat"):
            user_input[feature] = column.number_input(f"{feature} (Categorical)", value=-1, step=1, min_value=-1)
        else:
            user_input[feature] = column.number_input(f"{feature} (Numerical)", value=0.0, step=0.01, format="%.4f")

    submitted = st.form_submit_button("Predict claim probability", type="primary")

if submitted:
    with st.spinner("Running prediction..."):
        try:
            prediction, probability = predict_with_fallback(user_input)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

    if prediction == 1:
        st.error("🚨 Prediction: claim expected (1)")
    else:
        st.success("✅ Prediction: no claim expected (0)")

    st.metric("Probability of claim", f"{probability:.2%}")
    st.caption("The model estimates the chance of a claim using the selected features and the trained pipeline.")