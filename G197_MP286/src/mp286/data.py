from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from .config import ROOT, load_params

TARGET = "wait_min"
LEAKAGE_COLUMNS = ["token_called_after_min"]
NUMERIC_FEATURES = ["patients_ahead", "doctors_on_duty", "arrival_hour", "avg_consult_min", "lab_tests_pending", "distance_km"]
CATEGORICAL_FEATURES = ["department", "wing"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def data_path(name: str) -> Path:
    p = load_params()["data"]
    return ROOT / p[name]

def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def clean_frame(df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
    out = df.copy()
    # Normalize text fields against clean later/holdout vocabulary.
    if "department" in out:
        mapping = {
            "General": "General", " General": "General", "General ": "General", "GENERAL": "General", "Gen.": "General",
            "Paediatrics": "Paediatrics", " Paediatrics": "Paediatrics", "Paediatrics ": "Paediatrics", "PAEDIATRICS": "Paediatrics", "Pae.": "Paediatrics",
            "Orthopaedics": "Orthopaedics", " Orthopaedics": "Orthopaedics", "Orthopaedics ": "Orthopaedics", "ORTHOPAEDICS": "Orthopaedics", "Ort.": "Orthopaedics",
        }
        normalized = out["department"].astype(str).str.strip()
        out["department"] = normalized.map(mapping).fillna(normalized.str.title())
    # Annexe recorded consultation seconds; convert to minutes.
    if "avg_consult_min" in out and "wing" in out:
        mask = out["wing"].eq("Annexe") & out["avg_consult_min"].gt(60)
        out.loc[mask, "avg_consult_min"] = out.loc[mask, "avg_consult_min"] / 60.0
    # -999 is an explicit missing sentinel for arrival_hour.
    if "arrival_hour" in out:
        out.loc[out["arrival_hour"].eq(-999), "arrival_hour"] = np.nan
        out["arrival_hour"] = out["arrival_hour"].fillna(out["arrival_hour"].median())
    return out

def load_clean(name: str) -> pd.DataFrame:
    return clean_frame(pd.read_csv(data_path(name)))

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    clean = clean_frame(df, include_target=False)
    return clean[FEATURES].copy()
