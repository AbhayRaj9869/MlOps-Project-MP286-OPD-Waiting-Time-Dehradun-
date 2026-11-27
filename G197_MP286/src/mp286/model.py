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

from .data import (
    FEATURES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    prepare_features
)


# =========================================================
# MODEL ARCHITECTURE AND PREPROCESSING
# =========================================================

def build_model(
    n_estimators=350,
    max_depth=8,
    min_samples_leaf=2,
    random_state=42
):
    """
    Construct the complete machine learning pipeline.

    The pipeline combines data preprocessing and a Random Forest
    regressor so that the same transformations are applied during
    both training and inference.

    Args:
        n_estimators: Number of decision trees in the forest.
        max_depth: Maximum depth allowed for individual trees.
        min_samples_leaf: Minimum number of samples required
                          at a leaf node.
        random_state: Seed used for reproducible training.

    Returns:
        Pipeline: Configured preprocessing and regression pipeline.
    """

    # -----------------------------------------------------
    # 1. Numerical Feature Preprocessing
    # -----------------------------------------------------

    # Missing numerical values are replaced using the median.
    # Median imputation is less sensitive to extreme values
    # compared to using the mean.
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])


    # -----------------------------------------------------
    # 2. Categorical Feature Preprocessing
    # -----------------------------------------------------

    # Replace missing categorical values with the most
    # frequently occurring value in the respective column.
    #
    # OneHotEncoder converts categorical values into numerical
    # representations that can be processed by the model.
    #
    # handle_unknown="ignore" prevents inference failures when
    # previously unseen categories appear in incoming data.
    categorical = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])


    # -----------------------------------------------------
    # 3. Combine Numerical and Categorical Transformations
    # -----------------------------------------------------

    # ColumnTransformer applies different preprocessing
    # strategies to the corresponding feature groups.
    #
    # Numerical columns use median imputation, while
    # categorical columns use imputation and one-hot encoding.
    prep = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES)
    ])


    # -----------------------------------------------------
    # 4. Initialize the Regression Model
    # -----------------------------------------------------

    # Random Forest is an ensemble learning algorithm that
    # combines predictions from multiple decision trees.
    #
    # Averaging individual tree predictions helps reduce
    # variance and improves generalization.
    #
    # n_jobs=-1 allows parallel execution across available CPUs.
    reg = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1
    )


    # -----------------------------------------------------
    # 5. Assemble the Complete Pipeline
    # -----------------------------------------------------

    # Keeping preprocessing and model training together ensures
    # consistent feature transformations during prediction.
    return Pipeline([
        ("preprocess", prep),
        ("model", reg)
    ])


# =========================================================
# MODEL TRAINING
# =========================================================

def fit_model(frame: pd.DataFrame, **kwargs):
    """
    Train the regression pipeline using the supplied dataset.

    Args:
        frame: Training DataFrame containing input features
               and the target column 'wait_min'.
        **kwargs: Optional hyperparameters forwarded to build_model.

    Returns:
        Pipeline: Fitted machine learning pipeline.
    """

    # Construct the model using the supplied configuration.
    model = build_model(**kwargs)

    # Prepare input columns before fitting.
    # The target variable represents the waiting time in minutes.
    model.fit(
        prepare_features(frame),
        frame["wait_min"]
    )

    return model


# =========================================================
# MODEL INFERENCE
# =========================================================

def predict(model, frame: pd.DataFrame):
    """
    Generate waiting-time predictions for incoming records.

    Args:
        model: Previously trained regression pipeline.
        frame: DataFrame containing the required input features.

    Returns:
        Array of predicted waiting times.
    """

    # Apply the same feature preparation logic used during training
    # and pass the transformed data through the fitted pipeline.
    return model.predict(
        prepare_features(frame)
    )


# =========================================================
# MODEL PERFORMANCE EVALUATION
# =========================================================

def mae(model, frame: pd.DataFrame):
    """
    Calculate Mean Absolute Error (MAE) on a labeled dataset.

    MAE measures the average absolute difference between
    actual and predicted waiting times.

    Lower MAE values indicate better prediction accuracy.

    Args:
        model: Trained regression pipeline.
        frame: Evaluation dataset containing actual wait_min values.

    Returns:
        float: Mean absolute prediction error.
    """

    # Compare actual waiting times against model predictions.
    # Convert the result to a standard Python float for
    # easier logging and serialization.
    return float(
        mean_absolute_error(
            frame["wait_min"],
            predict(model, frame)
        )
    )


# =========================================================
# MODEL PERSISTENCE
# =========================================================

def save_model(model, path: str | Path):
    """
    Save a trained model to the specified filesystem location.

    Creates the parent directory automatically if it does
    not already exist.

    Args:
        model: Trained model or sklearn pipeline.
        path: Destination path for the serialized model.
    """

    # Ensure the destination directory is available.
    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Serialize the fitted pipeline using Joblib.
    joblib.dump(model, path)


def load_model(path: str | Path):
    """
    Load a previously serialized model from disk.

    Args:
        path: Location of the saved model artifact.

    Returns:
        The deserialized trained model.
    """

    return joblib.load(path)

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

