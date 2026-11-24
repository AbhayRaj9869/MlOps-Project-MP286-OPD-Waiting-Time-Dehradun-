# G197 — MP286 OPD Waiting Time (Dehradun)

SCSE3040 MLOps mini project for Group G197. The service predicts `wait_min` for Dehradun District Hospital OPD tokens. The pipeline includes seven data-quality checks, reproducible cleaning, MLflow experiment tracking and model registry, a FastAPI service, Docker/Kubernetes deployment files, GitHub Actions tests, JSON logging, drift monitoring, retraining, and final holdout predictions.

## Members
- S24CSEU1448 — Abhay RajYadav
- S24CSEU1460 — MayankTripathi
- S24CSEU1457 — ParthGaur

## Windows / PowerShell setup (Python 3.11)
From this repository folder run:

```powershell
.\setup.ps1
```

Activate the environment in later terminals:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Reproduce the pipeline

```powershell
python -m src.mp286.quality
python scripts/train.py
pytest -q
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for Swagger UI. `GET /health` returns `{"status":"ok"}`. `POST /predict` accepts exactly the eight prediction-time fields in `PROBLEM.md`; the after-event leakage column `token_called_after_min` is excluded.

## Docker
The assignment handbook specifies the `python:3.12-slim` base image for the container. The local development environment and CI use Python 3.11; the pinned dependencies support both versions. Build and run the required image:

```powershell
docker build -t mp286-api:1.0 .
docker run --rm -p 8000:8000 mp286-api:1.0
```

## Data and model artifacts
The original data are in `data/raw/`; DVC pointer files sit beside them. Configure a shared DVC remote using `DVC_SETUP.md` before relying on a fresh clone to retrieve the CSVs. `params.yaml` is the single source of dataset and model settings. The trained champion is `models/model.joblib`; the final 250-row holdout output is `predictions/holdout_predictions.csv` with exactly `record_id,prediction`. Training refreshes `reports/results.yaml` and the submission `findings.yaml`. The bundled MLflow database was generated on a Linux workspace; the training script backs it up as `mlflow.imported-backup.db` and creates a machine-local tracking database when it detects those stale absolute artifact paths.
