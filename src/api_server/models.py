from pydantic import BaseModel


class CustomerData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: float
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: int
    campaign: int
    pdays: int
    previous: int
    poutcome: str


class HealthCheckResult(BaseModel):
    status: str


class PredictionResult(BaseModel):
    status: str
    prediction: str
