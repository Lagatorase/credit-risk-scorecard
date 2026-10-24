"""FastAPI service for the credit risk scorecard.

  uvicorn api.main:app --port 8000

POST /score       PD, risk band, decision, reason codes and explanations for one applicant
GET  /health      whether the model is loaded, and which LLM backend is in use
GET  /model-card  docs/model_card.md as markdown
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from scorecard.config import get_settings
from scorecard.service import InputError, ScoringService, model_card_text

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # the model and its reference data load once at startup; a missing model shows up in /health
    try:
        state["service"] = ScoringService.from_settings(get_settings())
    except FileNotFoundError as exc:
        state["error"] = str(exc)
    yield
    state.clear()


app = FastAPI(title="Credit Risk Scorecard", version="1.0.0", lifespan=lifespan,
              description="Explainable PD model on Home Credit Default Risk, with reason codes and checked explanations.")


class ScoreRequest(BaseModel):
    applicant: dict[str, float | str | None] = Field(
        description="Model inputs by feature name (see config/model_features.txt). Missing inputs are treated as missing.",
        examples=[{"AMT_CREDIT": 450000, "AMT_ANNUITY": 24000, "AMT_INCOME_TOTAL": 180000, "EXT_SOURCE_2": 0.55}],
    )
    explanations: bool = Field(True, description="Also write the officer and customer explanations")
    include_shap: bool = Field(False, description="Also return every feature's SHAP value")


def _service() -> ScoringService:
    if "service" not in state:
        raise HTTPException(503, f"model not loaded: {state.get('error', 'unknown error')}")
    return state["service"]


@app.get("/health")
def health() -> dict:
    service = state.get("service")
    return {
        "status": "ok" if service else "model not loaded",
        "model_loaded": service is not None,
        "n_features": len(service.features) if service else 0,
        "llm_backend": (service.llm.name if service.llm else "none") if service else None,
        "customer_explanation": ("llm" if service.customer_llm else "template") if service else None,
    }


@app.get("/model-card", response_class=PlainTextResponse)
def model_card() -> PlainTextResponse:
    return PlainTextResponse(model_card_text(), media_type="text/markdown")


@app.post("/score")
def score(request: ScoreRequest) -> dict:
    service = _service()
    try:
        return service.score(request.applicant, explanations=request.explanations, include_shap=request.include_shap)
    except InputError as exc:
        raise HTTPException(422, str(exc)) from exc
