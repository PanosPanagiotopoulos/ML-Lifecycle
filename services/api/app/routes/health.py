from fastapi import APIRouter
from deps import get_inference_service

router = APIRouter()

@router.get("/health")
def health() -> dict:
    try:
        svc = get_inference_service()
        return {
            "status": "ok",
            "model_ready": svc.model_is_ready(),
            "retrieval_chunks": svc.chunk_count(),
        }
    except Exception:
        return {
            "status": "degraded",
            "model_ready": False,
            "retrieval_chunks": 0,
            "detail": "service_unavailable",
        }
