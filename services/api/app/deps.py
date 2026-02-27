from functools import lru_cache
from app.config import ApiConfig
from app.services.llm_model_service import LLMModelService

@lru_cache
def get_config() -> ApiConfig:
    return ApiConfig.load()

@lru_cache
def get_llm_model_service() -> LLMModelService:
    cfg = get_config()
    return LLMModelService(cfg)
