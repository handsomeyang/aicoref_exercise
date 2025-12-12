import joblib
from typing import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI
import pandas as pd
from utils import get_artifacts_dir
from models import CustomerData, HealthCheckResult, PredictionResult


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.best_ml_pipeline = joblib.load(
        get_artifacts_dir() / "best_ml_pipeline.joblib"
    )

    yield

    if app.state.best_ml_pipeline is not None:
        app.state.best_ml_pipeline = None


app = FastAPI(
    title="Term deposit subscription API",
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

    input_df = pd.DataFrame([data_dict])
    subscription_prob = app.state.best_ml_pipeline.predict_proba(input_df)[0][1]

    return PredictionResult(
        status="Success", prediction="yes" if subscription_prob > 0.5 else "no"
    )
