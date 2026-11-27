from __future__ import annotations

import pandas as pd
from .data import data_path


def inspect_data(train: pd.DataFrame, later: pd.DataFrame, holdout: pd.DataFrame) -> dict:
    # Initialize a dictionary to store all detected data quality issues.
    result = {}

    # Check whether the column "token_called_after_min" is missing from the holdout dataset.
    # Its absence is considered a potential indicator of data leakage prevention.
    result["leaking_column"] = {
        "present": "token_called_after_min" not in holdout.columns,
        "column": "token_called_after_min" if "token_called_after_min" not in holdout.columns else None
    }

    # Initialize the unit mix-up detection result.
    # This will be updated if a department/wing has unusually large consultation values.
    result["unit_mixup"] = {
        "present": False,
        "column": None,
        "group_value": None,
        "factor": None
    }

    # Calculate the median consultation time for each wing.
    by = train.groupby("wing")["avg_consult_min"].median()

    # Check whether the Annexe wing has consultation values significantly higher
    # than the median of all other wings.
    # A difference greater than 20 times may indicate seconds were recorded instead of minutes.
    if "Annexe" in by.index and by["Annexe"] > 20 * by.drop("Annexe").median():
        result["unit_mixup"] = {
            "present": True,
            "column": "avg_consult_min",
            "group_value": "Annexe",
            "factor": 60
        }

    # Identify rows where arrival_hour contains the sentinel value -999.
    # Sentinel values are commonly used to represent missing or invalid data.
    sentinel = train["arrival_hour"].eq(-999)

    # Record whether the sentinel value exists, the affected column,
    # the sentinel value itself, and the number of affected rows.
    result["sentinel_value"] = {
        "present": bool(sentinel.any()),
        "column": "arrival_hour" if sentinel.any() else None,
        "value": -999 if sentinel.any() else None,
        "count": int(sentinel.sum()) if sentinel.any() else None
    }

    # Count the number of completely duplicated rows in the training dataset.
    dup = int(train.duplicated().sum())

    # Record whether duplicate rows exist and their total count.
    result["duplicate_rows"] = {
        "present": dup > 0,
        "count": dup if dup > 0 else None
    }

    # Sort the training dataset chronologically by date.
    # This is necessary to detect consecutive periods where a sensor
    # repeatedly reports the same value.
    ordered = train.sort_values("date")

    # Initialize variables to track the longest consecutive run of identical values.
    # best stores: run length, column name, starting date, and ending date.
    best = (0, None, None, None)

    # Iterate through every numeric column in the training dataset.
    for col in train.select_dtypes(include="number").columns:

        # Start checking the sequence from the first row.
        start = 0

        # Convert the current column's values into a list for easier comparison.
        vals = ordered[col].tolist()

        # Iterate through the values, including one extra iteration
        # to ensure the final run is processed.
        for i in range(1, len(vals) + 1):

            # Check whether the current value differs from the value
            # at the beginning of the current consecutive run.
            # Also handle the end of the list.
            if i == len(vals) or vals[i] != vals[start]:

                # Calculate the length of the current consecutive run.
                # If it is longer than the previously recorded run,
                # update the best run's details.
                if i - start > best[0]:
                    best = (
                        i-start,
                        col,
                        ordered.iloc[start]["date"],
                        ordered.iloc[i-1]["date"]
                    )

                # Start a new consecutive run from the current position.
                start = i

    # Consider a sensor stuck if the same value appears continuously
    # for at least 30 consecutive rows.
    stuck = best[0] >= 30

    # Record whether a stuck sensor was detected.
    # If detected, store the column and the date range of the constant values.
    result["stuck_sensor"] = {
        "present": stuck,
        "column": best[1] if stuck else None,
        "start_date": best[2] if stuck else None,
        "end_date": best[3] if stuck else None
    }

    # Extract the target variable containing waiting times in minutes.
    target = train["wait_min"]

    # Identify potential target-value typos.
    # Values greater than five times the 99th percentile are considered suspicious.
    typo = target.gt(target.quantile(.99) * 5)

    # Record whether suspicious target values exist and their count.
    result["target_typos"] = {
        "present": bool(typo.any()),
        "count": int(typo.sum()) if typo.any() else None
    }

    # Extract the unique department names from the later dataset,
    # ignoring missing values.
    levels = set(later["department"].dropna().unique())

    # Check whether the training dataset contains more unique department names
    # than the later dataset. This may indicate spelling inconsistencies
    # or variations in department naming.
    variants = train["department"].nunique() > len(levels)

    # Record whether spelling variants may exist.
    # If detected, store the affected column and the number of true department levels.
    result["spelling_variants"] = {
        "present": variants,
        "column": "department" if variants else None,
        "true_levels": int(len(levels)) if variants else None
    }

    # Return the complete dictionary containing all detected data quality findings.
    return result


def findings(train_path=None, later_path=None, holdout_path=None):
    # Load the training dataset.
    # Use the provided file path if available; otherwise, use the default data path.
    train = pd.read_csv(train_path or data_path("train"))

    # Load the later dataset using the provided path or default location.
    later = pd.read_csv(later_path or data_path("later"))

    # Load the holdout dataset using the provided path or default location.
    holdout = pd.read_csv(holdout_path or data_path("holdout"))

    # Pass all three datasets to the inspection function and return its findings.
    return inspect_data(train, later, holdout)


if __name__ == "__main__":
    # Execute the findings function when this file is run directly.
    # pprint is used to display the resulting dictionary in a readable format.
    import pprint

    pprint.pp(findings())
