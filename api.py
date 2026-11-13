from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model_pipeline.pkl"
FEATURES_PATH = BASE_DIR / "models" / "selected_features.pkl"

app = FastAPI(title="Safe Driver Prediction API")
model_pipeline = None
selected_features: List[str] = []


def ensure_model_artifacts() -> None:
    global model_pipeline, selected_features
    if MODEL_PATH.exists() and FEATURES_PATH.exists():
        model_pipeline = joblib.load(MODEL_PATH)
        selected_features = joblib.load(FEATURES_PATH)
        return

    try:
        from train import train_model

        train_model()
        model_pipeline = joblib.load(MODEL_PATH)
        selected_features = joblib.load(FEATURES_PATH)
    except Exception as exc:
        raise RuntimeError(f"Could not load or train the model: {exc}") from exc


try:
    ensure_model_artifacts()
except RuntimeError:
    model_pipeline = None
    selected_features = []


@app.on_event("startup")
def startup_event() -> None:
    global model_pipeline, selected_features
    try:
        ensure_model_artifacts()
    except RuntimeError:
        model_pipeline = None
        selected_features = []


class PredictionRequest(BaseModel):
    features: Dict[str, Any]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/features", response_model=List[str])
def get_features() -> List[str]:
    if not selected_features:
        raise HTTPException(status_code=500, detail="Features list not loaded.")
    return selected_features


@app.post("/predict")
def predict(request: PredictionRequest):
    if not model_pipeline:
        raise HTTPException(status_code=500, detail="Model pipeline is not loaded.")

    try:
        normalized = {}
        for feature in selected_features:
            if feature not in request.features:
                raise KeyError(feature)
            value = request.features[feature]
            normalized[feature] = float(value)

        input_data = pd.DataFrame([normalized], columns=selected_features)
        probability = model_pipeline.predict_proba(input_data)[0, 1]
        prediction = int(model_pipeline.predict(input_data)[0])

        return {"prediction": prediction, "probability": float(probability)}
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing required feature: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc