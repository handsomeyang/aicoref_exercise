import joblib
from typing import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI
from utils import get_artifacts_dir


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


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict_subscription() -> None:
    pass
