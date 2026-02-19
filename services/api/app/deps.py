from functools import lru_cache
from config import ApiConfig
from services.inference_service import InferenceService

@lru_cache
def get_config() -> ApiConfig:
    return ApiConfig.load()

@lru_cache
def get_inference_service() -> InferenceService:
    cfg = get_config()
    return InferenceService(cfg)
