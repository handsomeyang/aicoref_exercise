import json
import joblib
from typing import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI
import pandas as pd
from utils import get_artifacts_dir, encode_binary_features
from .models import CustomerData, HealthCheckResult, PredictionResult


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    pipeline_path = get_artifacts_dir() / "best_ml_pipeline.joblib"
    try:
        app.state.ml_pipeline = joblib.load(pipeline_path)
    except FileNotFoundError:
        print(
            f"ERROR: Pipeline file not found at {pipeline_path}. Check deployment path."
        )
        app.state.ml_pipeline = None

    training_features_path = get_artifacts_dir() / "training_features.json"
    try:
        with open(training_features_path, "r") as f:
            app.state.training_features = json.load(f)
    except FileNotFoundError:
        print(
            f"ERROR: Training features file not found at {training_features_path}. Loading default training features."
        )
        app.state.training_features = [
            "age",
            "job",
            "marital",
            "education",
            "default",
            "balance",
            "housing",
            "loan",
            "contact",
            "day",
            "month",
            "duration",
            "campaign",
            "pdays",
            "previous",
            "poutcome",
        ]

    binary_features_path = get_artifacts_dir() / "binary_features.json"
    try:
        with open(binary_features_path, "r") as f:
            app.state.binary_features = json.load(f)
    except FileNotFoundError:
        print(
            f"ERROR: Binary features file not found at {binary_features_path}. Loading default binary features."
        )
        app.state.binary_features = ["default", "housing", "loan"]

    yield

    if app.state.ml_pipeline is not None:
        app.state.ml_pipeline = None


app = FastAPI(
    title="Term subscription prediction API",
    description="Predicts whether a customer will subscribe to a term deposit",
    version="0.1.0",
    contact={"email": "handsomeyang@gmail.com"},
    lifespan=lifespan,
)


@app.get("/", response_model=HealthCheckResult)
def health_check() -> HealthCheckResult:
    return HealthCheckResult(status="ok")


@app.post("/predict")
async def predict_subscription(data: CustomerData) -> PredictionResult:
    data_dict = data.model_dump()

    input_df = pd.DataFrame([data_dict], columns=app.state.training_features)
    encode_binary_features(input_df, app.state.binary_features)
    subscription_prob = app.state.ml_pipeline.predict_proba(input_df)[0][1]

    return PredictionResult(
        status="Success", prediction="yes" if subscription_prob > 0.5 else "no"
    )
