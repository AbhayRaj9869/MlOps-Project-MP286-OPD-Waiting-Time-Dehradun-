import pandas as pd
from src.mp286.data import clean_frame
from src.mp286.quality import inspect_data

def test_cleaning_normalizes_department_and_units():
    df=pd.DataFrame({"wing":["Annexe","New Wing"],"avg_consult_min":[600.0,10.0],"department":[" Gen.","General"],"arrival_hour":[-999,10]})
    out=clean_frame(df,include_target=False)
    assert out.loc[0,"avg_consult_min"] == 10
    assert out.loc[0,"department"] == "General"
    assert out.loc[0,"arrival_hour"] == 10

def test_quality_finds_required_injected_issues():
    train=pd.DataFrame({"date":["2026-01-01","2026-01-02"],"wing":["New Wing","New Wing"],"avg_consult_min":[10,11],"arrival_hour":[-999,10],"wait_min":[20,21],"department":["General","General"],"token_called_after_min":[20,21]})
    later=train.copy(); holdout=train.drop(columns=["token_called_after_min","wait_min"])
    result=inspect_data(train,later,holdout)
    assert result["leaking_column"]["present"] is True
    assert result["sentinel_value"]["count"] == 1
