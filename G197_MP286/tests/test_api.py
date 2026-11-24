from fastapi.testclient import TestClient
import app.main as api

client = TestClient(api.app)

def valid_payload():
    return {
        "patients_ahead": 20, "doctors_on_duty": 3, "arrival_hour": 10,
        "avg_consult_min": 10, "lab_tests_pending": 2, "distance_km": 4,
        "department": "General", "wing": "New Wing",
    }

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_schema_rejects_client_rules():
    payload = valid_payload()
    payload["patients_ahead"] = 151
    assert client.post("/predict", json=payload).status_code == 422
    payload = valid_payload()
    payload["lab_tests_pending"] = -1
    assert client.post("/predict", json=payload).status_code == 422

def test_schema_rejects_extra_fields():
    payload = valid_payload()
    payload["token_called_after_min"] = 20
    assert client.post("/predict", json=payload).status_code == 422

def test_predict_uses_shared_feature_preparation(monkeypatch):
    class FakeModel:
        def predict(self, frame):
            # The shared preprocessing function must provide exactly the model features.
            assert "token_called_after_min" not in frame.columns
            assert frame.loc[0, "department"] == "General"
            return [53.69]

    monkeypatch.setattr(api, "model", FakeModel())
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200
    assert abs(response.json()["prediction"] - 53.69) < 1e-8
