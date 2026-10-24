"""Tests for the FastAPI endpoints, with the small made-up model in place of the real one."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.main as api
from scorecard.service import ScoringService


@pytest.fixture
def client(tiny_explainer):
    # I put the service in directly instead of running the startup hook, which loads the real model
    api.state.clear()
    api.state["service"] = ScoringService(tiny_explainer, llm=None)
    yield TestClient(api.app)
    api.state.clear()


def test_health_reports_the_model_and_the_customer_default(client):
    body = client.get("/health").json()
    assert body["model_loaded"] and body["n_features"] == 153
    assert body["llm_backend"] == "none" and body["customer_explanation"] == "template"


def test_score_returns_pd_decision_reasons_and_labelled_explanations(client, applicant):
    body = client.post("/score", json={"applicant": applicant}).json()
    assert 0 < body["pd"] < 1 and body["decision"] in {"approve", "review", "decline"}
    assert body["reasons_officer"][0]["text"]
    assert body["explanations"]["officer"]["label"] and body["explanations"]["customer"]["source"] == "template"


def test_score_can_skip_explanations_and_add_shap(client, applicant):
    body = client.post("/score", json={"applicant": applicant, "explanations": False, "include_shap": True}).json()
    assert "explanations" not in body and len(body["shap"]) == 153


def test_bad_inputs_get_a_422_with_the_reason(client, applicant):
    response = client.post("/score", json={"applicant": {**applicant, "SHOE_SIZE": 42}})
    assert response.status_code == 422 and "SHOE_SIZE" in response.json()["detail"]


def test_a_missing_model_gives_503_not_a_crash(applicant):
    api.state.clear()
    api.state["error"] = "models/final_model.joblib not found"
    client = TestClient(api.app)
    assert client.get("/health").json()["model_loaded"] is False
    assert client.post("/score", json={"applicant": applicant}).status_code == 503
    api.state.clear()


def test_model_card_is_served_as_markdown(client):
    response = client.get("/model-card")
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/markdown")
