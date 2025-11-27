"""Configuration and environment variables management."""
import os
from pathlib import Path
from typing import Optional


class Config:
    """Application configuration."""
    
    # Directories
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    MODEL_DIR = ARTIFACTS_DIR / "model"
    MLRUNS_DIR = BASE_DIR / "mlruns"
    CONFIGS_DIR = BASE_DIR / "configs"
    
    # Logging
    LOG_DIR = BASE_DIR / "logs"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = LOG_DIR / "mllifecycle.log"
    
    # PDF Processing
    PDF_CHUNK_SIZE = int(os.getenv("PDF_CHUNK_SIZE", "512"))
    PDF_CHUNK_OVERLAP = int(os.getenv("PDF_CHUNK_OVERLAP", "50"))
    PDF_MIN_CHUNK_SIZE = int(os.getenv("PDF_MIN_CHUNK_SIZE", "50"))
    
    # Training
    TEST_SPLIT = float(os.getenv("TEST_SPLIT", "0.2"))
    RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
    
    # MLflow
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", str(MLRUNS_DIR))
    MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "pdf_llm_finetuning")
    
    # API
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"
    
    # Model serving
    MODEL_CONFIG_PATH = MODEL_DIR / "config.json"
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        for directory in [cls.DATA_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR,
                         cls.ARTIFACTS_DIR, cls.MODEL_DIR, cls.MLRUNS_DIR,
                         cls.LOG_DIR, cls.CONFIGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_config_dict(cls) -> dict:
        """Get all configuration as dictionary."""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith('_') and key.isupper()
        }