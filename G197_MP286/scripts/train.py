from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mlflow
from mlflow.tracking import MlflowClient
from src.mp286.config import ROOT, load_params
from src.mp286.data import load_clean, sha256_file
from src.mp286.model import fit_model, mae, predict, save_model
from src.mp286.monitor import psi_all, mae_by_group
import yaml, pandas as pd, sqlite3, shutil

def prepare_portable_mlflow_db():
    """Back up a DB whose artifact URIs point to the original Linux workspace.

    The bundled DB was produced in a different machine's absolute path. A fresh
    local DB lets MLflow create artifact locations that are valid on this machine.
    """
    db_path = ROOT / "mlflow.db"
    if not db_path.exists():
        return
    try:
        with sqlite3.connect(db_path) as connection:
            locations = [row[0] for row in connection.execute("SELECT artifact_location FROM experiments")]
    except sqlite3.Error:
        locations = []
    if any("/home/ubuntu/" in str(location) for location in locations):
        backup = ROOT / "mlflow.imported-backup.db"
        if not backup.exists():
            shutil.copy2(db_path, backup)
        for suffix in ["", "-wal", "-shm"]:
            candidate = Path(str(db_path) + suffix)
            if candidate.exists():
                candidate.unlink()
        print(f"Backed up non-portable MLflow DB to {backup.name}; creating a local tracking DB.")

def main():
    params=load_params(); train=load_clean("train"); later=load_clean("later"); holdout=load_clean("holdout")
    prepare_portable_mlflow_db()
    mlflow.set_tracking_uri("sqlite:///mlflow.db"); mlflow.set_experiment("MP286_OPD_Wait_Time")
    data_hash=sha256_file(ROOT/params["data"]["train"])
    runs=[]
    settings=[(180,6,2),(350,8,2),(500,10,1)]
    for idx,(n,d,leaf) in enumerate(settings,1):
        with mlflow.start_run(run_name=f"v1_candidate_{idx}") as run:
            m=fit_model(train,n_estimators=n,max_depth=d,min_samples_leaf=leaf,random_state=42)
            score=mae(m,later)
            mlflow.log_params({"data_sha256":data_hash,"model":"RandomForestRegressor","n_estimators":n,"max_depth":d,"min_samples_leaf":leaf})
            mlflow.log_metric("mae_later",score)
            runs.append((score,run.info.run_id,m))
    runs.sort(key=lambda x:x[0]); v1_score,v1_run,v1=runs[0]
    # Register only the selected v1 after comparing all three candidates.
    with mlflow.start_run(run_name="v1_selected") as run:
        mlflow.log_params({"data_sha256":data_hash,"selected_candidate_run":v1_run})
        mlflow.log_metric("mae_later",v1_score)
        mlflow.sklearn.log_model(v1,"model",registered_model_name="mp286-wait-model",serialization_format="cloudpickle")
        v1_run=run.info.run_id
    # Train v2 on first 70% of later, ordered by date, and register it.
    ordered=later.sort_values("date").reset_index(drop=True); split=int(len(ordered)*.7)
    train_v2=pd.concat([train,ordered.iloc[:split]],ignore_index=True); check=ordered.iloc[split:]
    with mlflow.start_run(run_name="v2_retrain") as run:
        v2=fit_model(train_v2,**params["model"]); v2_score=mae(v2,check)
        mlflow.log_params({"data_sha256":data_hash,"model":"RandomForestRegressor","training_rows":len(train_v2),"retrain_split":0.7})
        mlflow.log_metric("mae_check",v2_score)
        mlflow.sklearn.log_model(v2,"model",registered_model_name="mp286-wait-model",serialization_format="cloudpickle")
        v2_run=run.info.run_id
    client=MlflowClient(); versions=client.search_model_versions("name='mp286-wait-model'")
    # v1 is the first registered version; v2 the later one. Promote only if >=5% better.
    v1_version=min(int(x.version) for x in versions if x.run_id==v1_run)
    v2_version=max(int(x.version) for x in versions if x.run_id==v2_run)
    promote=v2_score <= v1_score*0.95
    champion_version=v2_version if promote else v1_version
    client.set_registered_model_alias("mp286-wait-model","champion",champion_version)
    champion=v2 if promote else v1; save_model(champion,ROOT/"models/model.joblib")
    pred=pd.DataFrame({"record_id":holdout["record_id"],"prediction":predict(champion,holdout)})
    (ROOT/"predictions").mkdir(exist_ok=True); pred.to_csv(ROOT/"predictions/holdout_predictions.csv",index=False)
    report={"tracking":{"experiment_name":"MP286_OPD_Wait_Time","best_run_id":v1_run,"registered_model_name":"mp286-wait-model","champion_version":int(champion_version)},"monitoring":{"psi":psi_all(train,later,10),"mae_by_group":mae_by_group(v1,later),"mae_v1_on_check":mae(v1,check),"mae_v2_on_check":v2_score,"decision":"promote" if promote else "keep"}}
    report["monitoring"]["most_drifted_feature"]=max(report["monitoring"]["psi"],key=report["monitoring"]["psi"].get); report["monitoring"]["worst_served_group"]=max(report["monitoring"]["mae_by_group"],key=report["monitoring"]["mae_by_group"].get)
    with open(ROOT/"reports/results.yaml","w",encoding="utf-8") as f:
        yaml.safe_dump(report,f,sort_keys=False)
    # Keep the submission form synchronized with the freshly computed results.
    form_path = ROOT / "findings.yaml"
    form = yaml.safe_load(form_path.read_text(encoding="utf-8")) if form_path.exists() else {"group_id": "G197", "problem_id": "MP286"}
    from src.mp286.quality import findings as compute_findings
    form["data_issues"] = compute_findings()
    form["tracking"] = report["tracking"]
    form["monitoring"] = report["monitoring"]
    form_path.write_text(yaml.safe_dump(form,sort_keys=False),encoding="utf-8")
    (ROOT / "reports" / "findings.yaml").write_text(form_path.read_text(encoding="utf-8"),encoding="utf-8")
    print(yaml.safe_dump(report,sort_keys=False))
if __name__=="__main__": main()
