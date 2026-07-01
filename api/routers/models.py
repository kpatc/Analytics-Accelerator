"""ML models router.

Endpoints that expose trained model metadata and predictions.
Full implementation arrives in Milestone 3.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ModelListResponse(BaseModel):
    """Schema for the model listing endpoint."""

    models: list[str]
    message: str


@router.get(
    "/models/list",
    response_model=ModelListResponse,
    summary="List available models",
)
async def list_models() -> ModelListResponse:
    """Return the list of trained ML models available for inference.

    Note: models have not been trained yet — run ``python scripts/train_models.py``
    to train the NovaMart model suite.
    """
    return ModelListResponse(
        models=[
            "churn_xgboost",
            "store_performance_lgbm",
            "price_elasticity_ols",
            "marketing_mix_mmm",
        ],
        message="Models not yet trained. Run: python scripts/train_models.py",
    )
