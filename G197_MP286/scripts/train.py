
from pathlib import Path
import sys
import mlflow
import yaml
import pandas as pd
import sqlite3
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlflow.tracking import MlflowClient
from src.mp286.config import ROOT, load_params
from src.mp286.data import load_clean, sha256_file
from src.mp286.model import fit_model, mae, predict, save_model
from src.mp286.monitor import psi_all, mae_by_group


# ---------------------------------------------------------
# MLflow Database Portability
# ---------------------------------------------------------

def prepare_portable_mlflow_db():
    """
    Check whether the existing MLflow database contains artifact
    paths from another machine and create a backup if necessary.
    """

    db_path = ROOT / "mlflow.db"

    if not db_path.exists():
        return

    try:
        with sqlite3.connect(db_path) as connection:
            locations = [
                row[0]
                for row in connection.execute(
                    "SELECT artifact_location FROM experiments"
                )
            ]

    except sqlite3.Error:
        locations = []

    # Replace machine-specific artifact references with local tracking.
    if any("/home/ubuntu/" in str(location) for location in locations):

        backup = ROOT / "mlflow.imported-backup.db"

        if not backup.exists():
            shutil.copy2(db_path, backup)

        # Remove the old database and its temporary journal files.
        for suffix in ["", "-wal", "-shm"]:
            candidate = Path(str(db_path) + suffix)

            if candidate.exists():
                candidate.unlink()

        print(
            f"Backed up non-portable MLflow DB to {backup.name}; "
            "creating a local tracking DB."
        )


# ---------------------------------------------------------
# Main Training and Evaluation Pipeline
# ---------------------------------------------------------

def main():

    # Load configuration and prepare the datasets.
    params = load_params()

    train = load_clean("train")
    later = load_clean("later")
    holdout = load_clean("holdout")

    # Initialize MLflow experiment tracking.
    prepare_portable_mlflow_db()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("MP286_OPD_Wait_Time")

    # Generate a checksum to identify the training dataset version.
    data_hash = sha256_file(ROOT / params["data"]["train"])

    runs = []

    # -----------------------------------------------------
    # Stage 1: Train and Compare Initial Model Candidates
    # -----------------------------------------------------

    settings = [
        (180, 6, 2),
        (350, 8, 2),
        (500, 10, 1)
    ]

    for idx, (n, d, leaf) in enumerate(settings, 1):

        with mlflow.start_run(run_name=f"v1_candidate_{idx}") as run:

            # Train a candidate using the selected hyperparameters.
            m = fit_model(
                train,
                n_estimators=n,
                max_depth=d,
                min_samples_leaf=leaf,
                random_state=42
            )

            # Evaluate the candidate using mean absolute error.
            score = mae(m, later)

            # Record the configuration and evaluation results.
            mlflow.log_params({
                "data_sha256": data_hash,
                "model": "RandomForestRegressor",
                "n_estimators": n,
                "max_depth": d,
                "min_samples_leaf": leaf
            })

            mlflow.log_metric("mae_later", score)

            runs.append((score, run.info.run_id, m))

    # Select the candidate with the lowest MAE.
    runs.sort(key=lambda x: x[0])

    v1_score, v1_run, v1 = runs[0]


    # -----------------------------------------------------
    # Stage 2: Register the Selected Initial Model (V1)
    # -----------------------------------------------------

    with mlflow.start_run(run_name="v1_selected") as run:

        mlflow.log_params({
            "data_sha256": data_hash,
            "selected_candidate_run": v1_run
        })

        mlflow.log_metric("mae_later", v1_score)

        # Register the selected model in the MLflow model registry.
        mlflow.sklearn.log_model(
            v1,
            "model",
            registered_model_name="mp286-wait-model",
            serialization_format="cloudpickle"
        )

        v1_run = run.info.run_id


    # -----------------------------------------------------
    # Stage 3: Retrain the Model Using Later Observations
    # -----------------------------------------------------

    # Sort observations chronologically to preserve temporal order.
    ordered = later.sort_values("date").reset_index(drop=True)

    split = int(len(ordered) * 0.7)

    # Combine the original training data with the first 70%
    # of the later observations.
    train_v2 = pd.concat(
        [train, ordered.iloc[:split]],
        ignore_index=True
    )

    # Reserve the remaining observations for evaluation.
    check = ordered.iloc[split:]

    with mlflow.start_run(run_name="v2_retrain") as run:

        v2 = fit_model(train_v2, **params["model"])

        v2_score = mae(v2, check)

        # Track retraining configuration and performance.
        mlflow.log_params({
            "data_sha256": data_hash,
            "model": "RandomForestRegressor",
            "training_rows": len(train_v2),
            "retrain_split": 0.7
        })

        mlflow.log_metric("mae_check", v2_score)

        # Register the retrained model as a new version.
        mlflow.sklearn.log_model(
            v2,
            "model",
            registered_model_name="mp286-wait-model",
            serialization_format="cloudpickle"
        )

        v2_run = run.info.run_id


    # -----------------------------------------------------
    # Stage 4: Compare Model Versions and Select Champion
    # -----------------------------------------------------

    client = MlflowClient()

    versions = client.search_model_versions(
        "name='mp286-wait-model'"
    )

    # Identify the registered versions corresponding to each run.
    v1_version = min(
        int(x.version)
        for x in versions
        if x.run_id == v1_run
    )

    v2_version = max(
        int(x.version)
        for x in versions
        if x.run_id == v2_run
    )

    # Promote V2 only when it achieves at least 5% improvement.
    promote = v2_score <= v1_score * 0.95

    champion_version = v2_version if promote else v1_version

    # Update the champion alias in the MLflow model registry.
    client.set_registered_model_alias(
        "mp286-wait-model",
        "champion",
        champion_version
    )

    champion = v2 if promote else v1

    # Persist the selected champion model locally.
    save_model(champion, ROOT / "models/model.joblib")


    # -----------------------------------------------------
    # Stage 5: Generate Predictions on Holdout Data
    # -----------------------------------------------------

    pred = pd.DataFrame({
        "record_id": holdout["record_id"],
        "prediction": predict(champion, holdout)
    })

    # Ensure the predictions directory exists before exporting.
    (ROOT / "predictions").mkdir(exist_ok=True)

    pred.to_csv(
        ROOT / "predictions/holdout_predictions.csv",
        index=False
    )


    # -----------------------------------------------------
    # Stage 6: Model and Data Drift Monitoring
    # -----------------------------------------------------

    report = {
        "tracking": {
            "experiment_name": "MP286_OPD_Wait_Time",
            "best_run_id": v1_run,
            "registered_model_name": "mp286-wait-model",
            "champion_version": int(champion_version)
        },

        "monitoring": {
            # Population Stability Index for feature drift.
            "psi": psi_all(train, later, 10),

            # Evaluate prediction errors across different groups.
            "mae_by_group": mae_by_group(v1, later),

            "mae_v1_on_check": mae(v1, check),
            "mae_v2_on_check": v2_score,

            "decision": "promote" if promote else "keep"
        }
    }

    # Identify the feature exhibiting the highest distribution drift.
    report["monitoring"]["most_drifted_feature"] = max(
        report["monitoring"]["psi"],
        key=report["monitoring"]["psi"].get
    )

    # Identify the group with the highest prediction error.
    report["monitoring"]["worst_served_group"] = max(
        report["monitoring"]["mae_by_group"],
        key=report["monitoring"]["mae_by_group"].get
    )


    # -----------------------------------------------------
    # Stage 7: Export Results and Synchronize Findings
    # -----------------------------------------------------

    # Save the complete experiment and monitoring report.
    with open(
        ROOT / "reports/results.yaml",
        "w",
        encoding="utf-8"
    ) as f:

        yaml.safe_dump(report, f, sort_keys=False)


    # Keep the submission form synchronized with the latest results.
    form_path = ROOT / "findings.yaml"

    form = (
        yaml.safe_load(form_path.read_text(encoding="utf-8"))
        if form_path.exists()
        else {"group_id": "G197", "problem_id": "MP286"}
    )

    from src.mp286.quality import findings as compute_findings

    form["data_issues"] = compute_findings()
    form["tracking"] = report["tracking"]
    form["monitoring"] = report["monitoring"]

    form_path.write_text(
        yaml.safe_dump(form, sort_keys=False),
        encoding="utf-8"
    )

    # Maintain a synchronized copy inside the reports directory.
    (ROOT / "reports" / "findings.yaml").write_text(
        form_path.read_text(encoding="utf-8"),
        encoding="utf-8"
    )

    # Display the final pipeline results.
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
