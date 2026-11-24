from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from .data import NUMERIC_FEATURES, prepare_features
from .model import predict

def psi(train: pd.DataFrame, later: pd.DataFrame, column: str, bins: int = 10) -> float:
    a = train[column].astype(float).to_numpy(); b = later[column].astype(float).to_numpy()
    edges = np.unique(np.nanquantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2: return 0.0
    edges[0] = -np.inf; edges[-1] = np.inf
    pa = np.histogram(a, edges)[0] / len(a); pb = np.histogram(b, edges)[0] / len(b)
    pa = np.clip(pa, 1e-6, None); pb = np.clip(pb, 1e-6, None)
    return float(np.sum((pb-pa) * np.log(pb/pa)))

def psi_all(train, later, bins=10): return {c: psi(train, later, c, bins) for c in NUMERIC_FEATURES}

def mae_by_group(model, frame, column="wing"):
    preds = predict(model, frame); tmp=frame.copy(); tmp["prediction"]=preds
    return {str(k): float(mean_absolute_error(g["wait_min"], g["prediction"])) for k,g in tmp.groupby(column)}
