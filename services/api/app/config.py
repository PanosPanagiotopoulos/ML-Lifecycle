import os
import json
import socket
from urllib.parse import urlparse, urlunparse
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
    "mlflow_tracking_uri": "MLFLOW_TRACKING_URI",
    "model_name": "MLFLOW_MODEL_NAME",
    "model_stage": "MLFLOW_MODEL_STAGE",
    "log_dir": "LOG_DIR",
    "processed_chunks_path": "PROCESSED_CHUNKS_PATH",
    "retrieval_top_k": "RETRIEVAL_TOP_K",
    "chat_default_max_tokens": "CHAT_DEFAULT_MAX_TOKENS",
    "chat_default_temperature": "CHAT_DEFAULT_TEMPERATURE",
}


def _resolve_tracking_uri(uri: str) -> str:
    parsed = urlparse(uri)
    host = parsed.hostname
    if host != "mlflow":
        return uri

    if os.path.exists("/.dockerenv"):
        return uri

    try:
        socket.gethostbyname("mlflow")
        return uri
    except OSError:
        port = parsed.port or 5000
        netloc = f"localhost:{port}"
        return urlunparse((parsed.scheme or "http", netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))

class ApiConfig(BaseModel):
    mlflow_tracking_uri: str
    model_name: str
    model_stage: str
    log_dir: str
    processed_chunks_path: str
    retrieval_top_k: int
    chat_default_max_tokens: int
    chat_default_temperature: float

    @classmethod
    def load(cls) -> "ApiConfig":
        env_file_path = os.getenv("API_ENV_FILE", DEFAULT_ENV_FILE)
        _load_env_file(env_file_path)

        config_json_path = os.getenv("API_CONFIG_JSON", DEFAULT_CONFIG_JSON)
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

        if "mlflow_tracking_uri" in resolved:
            resolved["mlflow_tracking_uri"] = _resolve_tracking_uri(str(resolved["mlflow_tracking_uri"]))

        missing = [k for k in cls.model_fields.keys() if k not in resolved]
        if missing:
            raise ValueError(f"Missing API config keys: {', '.join(missing)}")

        return cls.model_validate(resolved)
