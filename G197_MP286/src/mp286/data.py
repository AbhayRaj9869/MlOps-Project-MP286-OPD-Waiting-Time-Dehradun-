
from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from .config import ROOT, load_params


# =========================================================
# DATASET CONFIGURATION
# =========================================================

# Target variable that the regression model attempts to predict.
# In this project, the objective is to estimate patient waiting
# time in minutes.
TARGET = "wait_min"


# Columns that must not be used as model inputs because they
# contain information that would not be available at prediction
# time, potentially introducing target leakage.
LEAKAGE_COLUMNS = [
    "token_called_after_min"
]


# Numerical input variables used for model training.
NUMERIC_FEATURES = [
    "patients_ahead",
    "doctors_on_duty",
    "arrival_hour",
    "avg_consult_min",
    "lab_tests_pending",
    "distance_km"
]


# Categorical variables representing non-numerical attributes.
CATEGORICAL_FEATURES = [
    "department",
    "wing"
]


# Combine both feature groups into a single ordered list.
# This list is used to maintain consistent input columns
# throughout training and inference.
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# =========================================================
# DATASET PATH MANAGEMENT
# =========================================================

def data_path(name: str) -> Path:
    """
    Resolve the location of a dataset using project configuration.

    Args:
        name: Dataset identifier defined in params.yaml
              (e.g. train, later, or holdout).

    Returns:
        Path: Absolute path to the requested dataset.
    """

    # Load dataset directory information from the YAML configuration.
    p = load_params()["data"]

    # Combine the project root with the configured dataset path.
    return ROOT / p[name]


# =========================================================
# DATA INTEGRITY AND CHECKSUM GENERATION
# =========================================================

def sha256_file(path: str | Path) -> str:
    """
    Generate a SHA-256 checksum for a file.

    The checksum can be used to identify a particular version
    of a dataset and verify data integrity.

    This is useful for experiment tracking and reproducibility,
    especially when multiple model versions are trained.

    Args:
        path: Location of the file to be hashed.

    Returns:
        str: Hexadecimal SHA-256 digest.
    """

    # Read the file contents and calculate its cryptographic hash.
    # Identical file contents produce the same checksum.
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


# =========================================================
# DATA CLEANING AND NORMALIZATION
# =========================================================

def clean_frame(
    df: pd.DataFrame,
    include_target: bool = True
) -> pd.DataFrame:
    """
    Clean and standardize the input DataFrame.

    Handles inconsistent categorical values, incorrect
    consultation-time units, and missing arrival-hour values.

    A copy is created to avoid modifying the original dataset.

    Args:
        df: Raw or partially processed input DataFrame.
        include_target: Indicates whether the target column
                        is expected to be present.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """

    # Work on a separate copy so the original input remains unchanged.
    out = df.copy()


    # -----------------------------------------------------
    # 1. Normalize Department Names
    # -----------------------------------------------------

    # Different spellings, abbreviations, and extra spaces
    # may represent the same hospital department.
    #
    # Standardizing these values prevents the model from
    # treating equivalent categories as separate features.

    if "department" in out:

        mapping = {
            "General": "General",
            " General": "General",
            "General ": "General",
            "GENERAL": "General",
            "Gen.": "General",

            "Paediatrics": "Paediatrics",
            " Paediatrics": "Paediatrics",
            "Paediatrics ": "Paediatrics",
            "PAEDIATRICS": "Paediatrics",
            "Pae.": "Paediatrics",

            "Orthopaedics": "Orthopaedics",
            " Orthopaedics": "Orthopaedics",
            "Orthopaedics ": "Orthopaedics",
            "ORTHOPAEDICS": "Orthopaedics",
            "Ort.": "Orthopaedics",
        }

        # Convert values to strings and remove leading/trailing
        # whitespace before applying the standardization mapping.
        normalized = out["department"].astype(str).str.strip()

        # Replace known variations with their canonical names.
        # Any unrecognized value is retained in title case.
        out["department"] = (
            normalized.map(mapping)
            .fillna(normalized.str.title())
        )


    # -----------------------------------------------------
    # 2. Standardize Consultation Duration Units
    # -----------------------------------------------------

    # Consultation duration is expected to be measured in minutes.
    #
    # Records belonging to the Annexe wing may contain values
    # recorded in seconds instead of minutes.
    #
    # Values above 60 are converted to minutes by dividing
    # the recorded duration by 60.

    if "avg_consult_min" in out and "wing" in out:

        mask = (
            out["wing"].eq("Annexe")
            & out["avg_consult_min"].gt(60)
        )

        out.loc[mask, "avg_consult_min"] = (
            out.loc[mask, "avg_consult_min"] / 60.0
        )


    # -----------------------------------------------------
    # 3. Handle Missing Arrival-Hour Values
    # -----------------------------------------------------

    # The value -999 is used as a special sentinel to represent
    # missing arrival-hour information.
    #
    # Sentinel values are converted to NaN so that they are
    # handled as missing observations rather than actual hours.

    if "arrival_hour" in out:

        out.loc[
            out["arrival_hour"].eq(-999),
            "arrival_hour"
        ] = np.nan

        # Replace missing arrival-hour values with the median
        # of the available observations in the current DataFrame.
        #
        # Median imputation provides a representative value
        # without being heavily influenced by extreme observations.
        out["arrival_hour"] = (
            out["arrival_hour"]
            .fillna(out["arrival_hour"].median())
        )


    # Return the cleaned dataset for downstream processing.
    return out


# =========================================================
# DATASET LOADING
# =========================================================

def load_clean(name: str) -> pd.DataFrame:
    """
    Load a dataset from its configured location and apply
    the standard cleaning operations.

    Args:
        name: Dataset identifier from params.yaml.

    Returns:
        pd.DataFrame: Loaded and cleaned dataset.
    """

    # Resolve the dataset path and load the CSV into memory.
    raw_data = pd.read_csv(data_path(name))

    # Apply normalization and missing-value handling before
    # returning the dataset to the training pipeline.
    return clean_frame(raw_data)


# =========================================================
# FEATURE PREPARATION
# =========================================================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and prepare the feature columns required by the model.

    Target and non-feature columns are excluded from the returned
    DataFrame. The selected columns follow the predefined order.

    Args:
        df: Input DataFrame containing raw or cleaned records.

    Returns:
        pd.DataFrame: Cleaned feature matrix.
    """

    # Apply the same cleaning operations used during dataset loading.
    # Target-independent transformations are performed before
    # selecting the model input columns.
    clean = clean_frame(df, include_target=False)

    # Retain only the approved numerical and categorical features.
    # A copy is returned to avoid unintended modifications
    # to the original DataFrame.
    return clean[FEATURES].copy()
