from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from .data import FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, prepare_features

def build_model(n_estimators=350, max_depth=8, min_samples_leaf=2, random_state=42):
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)])
    reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state, n_jobs=-1)
    return Pipeline([("preprocess", prep), ("model", reg)])

def fit_model(frame: pd.DataFrame, **kwargs):
    model = build_model(**kwargs)
    model.fit(prepare_features(frame), frame["wait_min"])
    return model

def predict(model, frame: pd.DataFrame):
    return model.predict(prepare_features(frame))

def mae(model, frame: pd.DataFrame):
    return float(mean_absolute_error(frame["wait_min"], predict(model, frame)))

def save_model(model, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True); joblib.dump(model, path)

def load_model(path: str | Path): return joblib.load(path)
