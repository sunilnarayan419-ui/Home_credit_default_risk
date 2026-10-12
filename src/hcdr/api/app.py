"""FastAPI application factory: ``uvicorn hcdr.api.app:app``.

The model artifact (``scripts/train_model.py``) is loaded once at startup and shared by all requests.
If it is missing or invalid the service still starts: ``/health`` reports ``model_loaded: false`` and the
prediction routes answer 503. Responses never contain file paths, stack traces or request payloads.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .. import __version__
from ..config import MODEL_PATH
from ..inference import ArtifactError, InferenceInputError, ModelBundle, load_bundle, predict_default_probability
from .schemas import (
    RAW_COLUMNS,
    ApplicationFeatures,
    BatchPredictionResponse,
    BatchRequest,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

log = logging.getLogger("hcdr.api")


def get_bundle(request: Request) -> ModelBundle:
    bundle = request.app.state.bundle
    if bundle is None:
        raise HTTPException(status_code=503, detail="Model is not available. Train and install the model artifact first.")
    return bundle


BundleDep = Annotated[ModelBundle, Depends(get_bundle)]


def _load(app: FastAPI, path: Path) -> None:
    app.state.bundle = None
    try:
        bundle = load_bundle(path)
        if set(bundle.raw_columns) != set(RAW_COLUMNS):
            raise ArtifactError("model artifact was trained on different raw columns than the API schema")
        app.state.bundle = bundle
        log.info("model artifact loaded (%s)", ", ".join(bundle.models))
    except ArtifactError as exc:
        log.warning("model artifact unavailable: %s", exc)


def create_app(model_path: Optional[Path] = None) -> FastAPI:
    path = Path(model_path) if model_path else MODEL_PATH

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _load(app, path)
        yield

    app = FastAPI(
        title="Home Credit default-risk API",
        version=__version__,
        description="Scores raw loan applications with the project's LightGBM/XGBoost/CatBoost blend. "
        "Scores are model outputs for ranking; they require independent validation before any credit decision.",
        lifespan=lifespan,
    )
    app.state.bundle = None

    unavailable = {503: {"model": ErrorResponse, "description": "Model artifact not loaded"}}

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    def health(request: Request) -> HealthResponse:
        loaded = request.app.state.bundle is not None
        return HealthResponse(status="ok" if loaded else "degraded", model_loaded=loaded)

    @app.get("/model-info", response_model=ModelInfoResponse, responses=unavailable, tags=["service"])
    def model_info(bundle: BundleDep) -> ModelInfoResponse:
        return ModelInfoResponse(**bundle.metadata)

    @app.post("/predict", response_model=PredictionResponse, responses=unavailable, tags=["inference"])
    def predict(application: ApplicationFeatures, bundle: BundleDep) -> PredictionResponse:
        p = predict_default_probability(bundle, pd.DataFrame([application.model_dump()]))
        return PredictionResponse(default_probability=float(p[0]))

    @app.post("/predict/batch", response_model=BatchPredictionResponse, responses=unavailable, tags=["inference"])
    def predict_batch(batch: BatchRequest, bundle: BundleDep) -> BatchPredictionResponse:
        p = predict_default_probability(bundle, pd.DataFrame([r.model_dump() for r in batch.records]))
        return BatchPredictionResponse(count=len(p), predictions=[PredictionResponse(default_probability=float(x)) for x in p])

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default body echoes the submitted values; financial inputs must not be reflected back.
        errors = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    @app.exception_handler(InferenceInputError)
    async def inference_input_error(_: Request, exc: InferenceInputError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled error on %s %s: %s", request.method, request.url.path, type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app


app = create_app()
