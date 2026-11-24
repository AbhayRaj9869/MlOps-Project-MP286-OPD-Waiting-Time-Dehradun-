from __future__ import annotations
import pandas as pd
from .data import data_path

def inspect_data(train: pd.DataFrame, later: pd.DataFrame, holdout: pd.DataFrame) -> dict:
    result = {}
    result["leaking_column"] = {"present": "token_called_after_min" not in holdout.columns, "column": "token_called_after_min" if "token_called_after_min" not in holdout.columns else None}
    # A 60x jump in Annexe consultation values identifies seconds vs minutes.
    result["unit_mixup"] = {"present": False, "column": None, "group_value": None, "factor": None}
    by = train.groupby("wing")["avg_consult_min"].median()
    if "Annexe" in by.index and by["Annexe"] > 20 * by.drop("Annexe").median():
        result["unit_mixup"] = {"present": True, "column": "avg_consult_min", "group_value": "Annexe", "factor": 60}
    sentinel = train["arrival_hour"].eq(-999)
    result["sentinel_value"] = {"present": bool(sentinel.any()), "column": "arrival_hour" if sentinel.any() else None, "value": -999 if sentinel.any() else None, "count": int(sentinel.sum()) if sentinel.any() else None}
    dup = int(train.duplicated().sum())
    result["duplicate_rows"] = {"present": dup > 0, "count": dup if dup > 0 else None}
    # Detect longest contiguous same-value run after sorting by date.
    ordered = train.sort_values("date")
    best = (0, None, None, None)
    for col in train.select_dtypes(include="number").columns:
        start = 0
        vals = ordered[col].tolist()
        for i in range(1, len(vals) + 1):
            if i == len(vals) or vals[i] != vals[start]:
                if i - start > best[0]: best = (i-start, col, ordered.iloc[start]["date"], ordered.iloc[i-1]["date"])
                start = i
    stuck = best[0] >= 30
    result["stuck_sensor"] = {"present": stuck, "column": best[1] if stuck else None, "start_date": best[2] if stuck else None, "end_date": best[3] if stuck else None}
    # A 10x typo is only counted when an upper-tail cluster is separated from the normal target range.
    target = train["wait_min"]
    typo = target.gt(target.quantile(.99) * 5)
    result["target_typos"] = {"present": bool(typo.any()), "count": int(typo.sum()) if typo.any() else None}
    levels = set(later["department"].dropna().unique())
    variants = train["department"].nunique() > len(levels)
    result["spelling_variants"] = {"present": variants, "column": "department" if variants else None, "true_levels": int(len(levels)) if variants else None}
    return result

def findings(train_path=None, later_path=None, holdout_path=None):
    train = pd.read_csv(train_path or data_path("train")); later = pd.read_csv(later_path or data_path("later")); holdout = pd.read_csv(holdout_path or data_path("holdout"))
    return inspect_data(train, later, holdout)


if __name__ == "__main__":
    import pprint
    pprint.pp(findings())
