"""ASGI composition root for the Vietnamese topic inference service.

HTTP routes live in route/, validation contracts in schema/, and the
trained-model registry/loading/inference logic in service/.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .route.health import router as health_router
from .route.model import router as model_router
from .route.prediction import router as prediction_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Vietnamese Topic Classification Inference API",
        version="0.2.0",
        description=(
            "Inference for trusted existing TF-IDF + Logistic Regression / LinearSVC "
            "experiments; no retraining or model uploads."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    app.include_router(health_router, prefix="/api", tags=["Health"])
    app.include_router(model_router, prefix="/api", tags=["Models"])
    app.include_router(prediction_router, prefix="/api", tags=["Prediction"])
    return app


app = create_app()
