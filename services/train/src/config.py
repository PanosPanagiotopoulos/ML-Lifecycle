import os
import json
from pydantic import BaseModel


BASE_DIR = os.path.dirname(__file__)
DEFAULT_CONFIG_JSON = os.path.abspath(os.path.join(BASE_DIR, "../config/settings.json"))
DEFAULT_ENV_FILE = os.path.abspath(os.path.join(BASE_DIR, "../.env.local"))


def _load_json_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_env_file(env_path: str) -> None:
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


ENV_ALIASES = {
    "raw_pdfs_dir": "RAW_PDFS_DIR",
    "processed_dir": "PROCESSED_DIR",
    "base_model_name": "BASE_MODEL_NAME",
    "registered_model_name": "MLFLOW_MODEL_NAME",
    "experiment_name": "MLFLOW_EXPERIMENT_NAME",
    "mlflow_tracking_uri": "MLFLOW_TRACKING_URI",
}

class TrainConfig(BaseModel):
    raw_pdfs_dir: str
    processed_dir: str

    base_model_name: str
    max_length: int
    chunk_size: int
    chunk_overlap: int

    lora_r: int
    lora_alpha: int
    lora_dropout: float

    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    warmup_ratio: float
    gradient_accumulation_steps: int
    eval_ratio: float
    seed: int

    mlflow_tracking_uri: str
    experiment_name: str
    registered_model_name: str

    @classmethod
    def load(cls) -> "TrainConfig":
        env_file_path = os.getenv("TRAIN_ENV_FILE", DEFAULT_ENV_FILE)
        _load_env_file(env_file_path)

        config_json_path = os.getenv("TRAIN_CONFIG_JSON", DEFAULT_CONFIG_JSON)
        json_cfg = _load_json_config(config_json_path)
        resolved: dict = {}
        for key, value in json_cfg.items():
            if value is not None:
                resolved[key] = value
        for key in cls.model_fields.keys():
            env_key = ENV_ALIASES.get(key, key.upper())
            env_value = os.getenv(env_key)
            if env_value is not None:
                resolved[key] = env_value

        missing = [k for k in cls.model_fields.keys() if k not in resolved]
        if missing:
            raise ValueError(f"Missing train config keys: {', '.join(missing)}")

        return cls.model_validate(resolved)
