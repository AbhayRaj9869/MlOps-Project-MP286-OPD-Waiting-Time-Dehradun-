from __future__ import annotations
import json, logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from src.mp286.model import load_model, predict as predict_frame
from src.mp286.data import ROOT

class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({"level": record.levelname, "message": record.getMessage()})
logger = logging.getLogger("mp286-api"); logger.setLevel(logging.INFO)
handler = logging.StreamHandler(); handler.setFormatter(JsonFormatter()); logger.handlers=[handler]
MODEL_PATH = ROOT / "models/model.joblib"
try:
    model = load_model(MODEL_PATH) if MODEL_PATH.exists() else None
except Exception:
    logger.exception("model_load_failed")
    model = None
app = FastAPI(title="MP286 OPD Wait Predictor", version="1.0.0")

class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patients_ahead: int = Field(..., ge=0, le=150)
    doctors_on_duty: int = Field(..., ge=1)
    arrival_hour: int = Field(..., ge=0, le=23)
    avg_consult_min: float = Field(..., gt=0)
    lab_tests_pending: int = Field(..., ge=0)
    distance_km: float = Field(..., ge=0)
    department: str
    wing: str

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    import pandas as pd
    if model is None: raise HTTPException(status_code=503, detail="model artifact is unavailable or incompatible")
    frame = pd.DataFrame([req.model_dump()])
    value = float(predict_frame(model, frame)[0])
    logger.info("prediction_served")
    return {"prediction": value}
