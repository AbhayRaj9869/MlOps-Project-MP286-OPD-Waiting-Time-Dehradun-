from __future__ import annotations

# Import NumPy for numerical operations, arrays, quantiles, histograms, and logarithms.
import numpy as np

# Import pandas for DataFrame-based data manipulation.
import pandas as pd

# Import Mean Absolute Error (MAE) for evaluating prediction errors.
from sklearn.metrics import mean_absolute_error

# Import the numeric feature list and feature-preparation function from the data module.
from .data import NUMERIC_FEATURES, prepare_features

# Import the prediction function from the model module.
from .model import predict


def psi(train: pd.DataFrame, later: pd.DataFrame, column: str, bins: int = 10) -> float:
    # Convert the selected column from both datasets to floating-point NumPy arrays.
    # 'train' represents the reference distribution, while 'later' represents
    # the distribution being compared against it.
    a = train[column].astype(float).to_numpy()
    b = later[column].astype(float).to_numpy()

    # Calculate quantile-based bin boundaries using the training dataset.
    # np.linspace(0, 1, bins + 1) creates evenly spaced quantile levels.
    # np.nanquantile ignores NaN values while calculating the quantiles.
    # np.unique removes duplicate boundaries, which can occur when values repeat.
    edges = np.unique(np.nanquantile(a, np.linspace(0, 1, bins + 1)))

    # If fewer than two unique boundaries exist, the column has no meaningful
    # variation for distribution comparison, so return PSI = 0.
    if len(edges) < 2:
        return 0.0

    # Replace the first and last bin boundaries with negative and positive infinity.
    # This ensures that all values, including values outside the original
    # training range, are included in the histogram bins.
    edges[0] = -np.inf
    edges[-1] = np.inf

    # Calculate the frequency distribution of the training data across the bins.
    # Divide the histogram counts by the total number of training observations
    # to convert counts into proportions.
    pa = np.histogram(a, edges)[0] / len(a)

    # Calculate the frequency distribution of the later dataset using the
    # same bin boundaries, ensuring that both distributions are comparable.
    pb = np.histogram(b, edges)[0] / len(b)

    # Clip the proportions to a minimum value of 1e-6.
    # This prevents zero probabilities, which would cause problems
    # when calculating logarithms.
    pa = np.clip(pa, 1e-6, None)
    pb = np.clip(pb, 1e-6, None)

    # Calculate the Population Stability Index (PSI).
    # PSI measures how much the distribution of a feature has shifted.
    # Formula:
    # PSI = sum((later_proportion - train_proportion) * log(later_proportion / train_proportion))
    # A larger PSI indicates a greater distributional change.
    return float(np.sum((pb-pa) * np.log(pb/pa)))


def psi_all(train, later, bins=10):
    # Calculate PSI for every numeric feature listed in NUMERIC_FEATURES.
    # Return a dictionary where each feature name maps to its PSI value.
    return {c: psi(train, later, c, bins) for c in NUMERIC_FEATURES}


def mae_by_group(model, frame, column="wing"):
    # Generate predictions for every row in the provided DataFrame
    # using the supplied trained model.
    preds = predict(model, frame)

    # Create a copy of the original DataFrame to avoid modifying it directly.
    tmp = frame.copy()

    # Add the model predictions as a new column named "prediction".
    tmp["prediction"] = preds

    # Group the dataset by the specified column, which defaults to "wing".
    # For each group, calculate the Mean Absolute Error between the actual
    # waiting times and the predicted waiting times.
    # Convert group names to strings and MAE values to floats.
    # Return the results as a dictionary.
    return {
        str(k): float(mean_absolute_error(g["wait_min"], g["prediction"]))
        for k, g in tmp.groupby(column)
    }
