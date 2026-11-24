# G197 Work Distribution

The supplied project files are prepared, but each member must make their own real commits from their own GitHub account and should be able to explain the whole pipeline in the viva. The assignments below divide primary ownership without making unsupported claims about work that a member did not perform.

## Abhay RajYadav — S24CSEU1448: data and model lead

Primary ownership: `src/mp286/data.py`, `src/mp286/quality.py`, `src/mp286/config.py`, `src/mp286/findings.py`, `scripts/train.py`, `src/mp286/model.py`, `reports/findings.yaml`, `reports/results.yaml`, `params.yaml`, and the final `predictions/holdout_predictions.csv`.

Abhay should explain the seven data checks, why `token_called_after_min` is leakage, why the Annexe `avg_consult_min` values are divided by 60, the `-999` sentinel, the stuck `lab_tests_pending` run, spelling normalization, the train/later split, MAE, Random Forest preprocessing, MLflow parameters and alias promotion rule. Abhay should make a small data-quality commit and a model/training commit under his own GitHub identity.

## Mayank Tripathi — S24CSEU1460: service and testing lead

Primary ownership: `app/main.py`, `tests/test_api.py`, `tests/test_data.py`, `Dockerfile`, `.dockerignore`, `requirements.txt`, and the API sections of `README.md`.

Mayank should explain the exact Pydantic request schema, `extra="forbid"`, `patients_ahead <= 150`, `lab_tests_pending >= 0`, `/health`, model loading, why personal details are not logged, JSON stdout logging, container layering, port 8000, and TestClient coverage. Mayank should make a service/test commit under his own GitHub identity and verify the CI test command locally.

## Parth Gaur — S24CSEU1457: operations and monitoring lead

Primary ownership: `.github/workflows/ci.yml`, `k8s/deployment.yaml`, `k8s/service.yaml`, `src/mp286/monitor.py`, `.dvc/config`, `data/raw/*.dvc`, and the deployment/monitoring sections of `README.md`.

Parth should explain DVC hashes, CI on push, Kubernetes selectors and two replicas, readiness versus liveness delays, PSI bins and interpretation, per-wing MAE, the 70/30 chronological retraining split, why v2 is promoted, and the final holdout artifact. Parth should make operations/monitoring commits under his own GitHub identity and confirm the GitHub Actions run is green.

## Shared responsibilities before submission

All three members should review every file, run `pytest -q`, inspect `/health` and `/predict`, understand the findings and promotion decision, and make at least one meaningful commit. The public repository URL must be identical in all three individual forms, and every member must fill the form separately before the deadline. Do not claim files or work in the form unless that member actually reviewed, changed, and can defend them.


## Suggested commit sequence (each person uses their own GitHub account)

1. **Abhay:** `data: document and test MP286 cleaning and quality findings`; then `model: reproduce MLflow v1/v2 and holdout predictions`.
2. **Mayank:** `api: enforce MP286 request schema and prediction endpoint`; then `test: cover health, validation and valid predictions`.
3. **Parth:** `ops: configure DVC pointers and CI`; then `deploy: verify Docker and Kubernetes manifests and monitoring`.

These are suggested commit subjects, not evidence that a commit has already been made. Each member must personally review and change the files they claim.
