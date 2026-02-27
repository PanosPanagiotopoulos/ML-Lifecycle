from fastapi import APIRouter
from app.deps import get_llm_model_service

router = APIRouter()

@router.get("/health")
def health() -> dict:
    try:
        llm_model_service = get_llm_model_service()
        return {
            "status": "ok",
            "model_ready": llm_model_service.model_is_ready(),
        }
    except Exception:
        return {
            "status": "degraded",
            "model_ready": False,
            "detail": "The service cannot access the configured model backend at this time.",
        }
