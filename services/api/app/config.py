import json
import os
import socket
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse, urlunparse

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, model_validator


APP_DIR = Path(__file__).resolve().parent
SERVICE_DIR = APP_DIR.parent
REPO_ROOT = SERVICE_DIR.parent.parent
DEFAULT_CONFIG_DIR = SERVICE_DIR / "config"
DEFAULT_DATA_ROOT = REPO_ROOT / "data"


ENV_ALIASES = {
    "mlflow_tracking_uri": "MLFLOW_TRACKING_URI",
    "model_name": "MLFLOW_MODEL_NAME",
    "model_stage": "MLFLOW_MODEL_STAGE",
    "model_source": "MODEL_SOURCE",
    "registry_model_uri": "REGISTRY_MODEL_URI",
    "local_model_dir": "LOCAL_MODEL_DIR",
    "log_dir": "LOG_DIR",
    "data_root": "DATA_ROOT",
    "chat_default_max_tokens": "CHAT_DEFAULT_MAX_TOKENS",
    "chat_default_temperature": "CHAT_DEFAULT_TEMPERATURE",
}


def _resolve_path(path_value: str, data_root: str) -> str:
    if not path_value:
        return path_value

    path_value = os.path.expandvars(path_value)
    path_value = os.path.expanduser(path_value)
    if os.path.isabs(path_value):
        return os.path.abspath(path_value)

    normalized = path_value.replace("\\", "/")
    if normalized.startswith("./data/"):
        return os.path.abspath(os.path.join(data_root, normalized[len("./data/"):]))
    if normalized == "./data":
        return os.path.abspath(data_root)
    if normalized.startswith("data/"):
        return os.path.abspath(os.path.join(data_root, normalized[len("data/"):]))
    if normalized == "data":
        return os.path.abspath(data_root)

    return os.path.abspath(os.path.join(str(REPO_ROOT), path_value))


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


def _load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    return loaded if isinstance(loaded, dict) else {}


def _normalize_environment_name(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    aliases = {
        "local": "development",
        "dev": "development",
        "docker": "production",
        "prod": "production",
    }
    return aliases.get(normalized, normalized)


def _detect_environment() -> str:
    explicit = os.getenv("APP_ENV") or os.getenv("API_ENV") or os.getenv("ENVIRONMENT")
    if explicit:
        return _normalize_environment_name(explicit)
    return "production" if Path("/.dockerenv").exists() else "development"


def _resolve_config_dir() -> Path:
    configured = os.getenv("API_CONFIG_DIR")
    return Path(configured).resolve() if configured else DEFAULT_CONFIG_DIR


def _resolve_env_file(environment: str) -> Path:
    configured = os.getenv("API_ENV_FILE")
    if configured:
        return Path(configured).resolve()

    candidates = [SERVICE_DIR / f".env.{environment}"]
    if environment == "development":
        candidates.append(SERVICE_DIR / ".env.local")
    elif environment == "production":
        candidates.append(SERVICE_DIR / ".env.docker")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return SERVICE_DIR / ".env.local"


def _load_layered_sources(environment: str) -> tuple[dict[str, Any], dict[str, str]]:
    explicit_json = os.getenv("API_CONFIG_JSON")
    if explicit_json:
        base_values = _load_json_file(Path(explicit_json).resolve())
        config_dir = Path(explicit_json).resolve().parent
    else:
        config_dir = _resolve_config_dir()
        base_values = _load_json_file(config_dir / "settings.json")
    env_settings_path = config_dir / f"settings.{environment}.json"
    if not env_settings_path.exists() and environment == "development":
        env_settings_path = config_dir / "settings.local.json"
    elif not env_settings_path.exists() and environment == "production":
        env_settings_path = config_dir / "settings.docker.json"

    env_values = _load_json_file(env_settings_path)
    merged_json = {**base_values, **env_values}

    env_file_values: dict[str, str] = {}
    env_file = _resolve_env_file(environment)
    if env_file.exists():
        raw = dotenv_values(env_file)
        env_file_values = {k: v for k, v in raw.items() if v is not None}

    return merged_json, env_file_values


def _none_if_blank(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized if normalized else None

class ApiConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mlflow_tracking_uri: str
    model_name: str
    model_stage: str
    model_source: Literal["registry", "local"] = "registry"
    registry_model_uri: str | None = None
    local_model_dir: str | None = None
    log_dir: str
    data_root: str = DEFAULT_DATA_ROOT
    chat_default_max_tokens: int
    chat_default_temperature: float

    @model_validator(mode="after")
    def _validate_source_fields(self) -> "ApiConfig":
        if self.model_source == "local" and not self.local_model_dir:
            raise ValueError("LOCAL_MODEL_DIR must be configured when MODEL_SOURCE is set to 'local'.")
        if self.model_source == "registry" and not self.model_name.strip():
            raise ValueError("MLFLOW_MODEL_NAME must be configured when MODEL_SOURCE is set to 'registry'.")
        return self

    @classmethod
    def load(cls) -> "ApiConfig":
        environment = _detect_environment()
        json_cfg, env_file_cfg = _load_layered_sources(environment)

        resolved: dict[str, Any] = {k: v for k, v in json_cfg.items() if v is not None}
        for key in cls.model_fields.keys():
            env_key = ENV_ALIASES.get(key, key.upper())
            env_value = env_file_cfg.get(env_key)
            if env_value is None:
                env_value = os.getenv(env_key)
            if env_value is not None:
                resolved[key] = env_value

        if "model_source" in resolved and resolved["model_source"]:
            resolved["model_source"] = str(resolved["model_source"]).strip().lower()

        resolved["registry_model_uri"] = _none_if_blank(resolved.get("registry_model_uri"))
        resolved["local_model_dir"] = _none_if_blank(resolved.get("local_model_dir"))

        if "mlflow_tracking_uri" in resolved:
            resolved["mlflow_tracking_uri"] = _resolve_tracking_uri(str(resolved["mlflow_tracking_uri"]))

        if "data_root" in resolved and resolved["data_root"]:
            resolved["data_root"] = _resolve_path(str(resolved["data_root"]), DEFAULT_DATA_ROOT)
        else:
            resolved["data_root"] = os.path.abspath(DEFAULT_DATA_ROOT)

        data_root = str(resolved["data_root"])
        if "log_dir" in resolved:
            resolved["log_dir"] = _resolve_path(str(resolved["log_dir"]), data_root)
        if "local_model_dir" in resolved and resolved["local_model_dir"]:
            resolved["local_model_dir"] = _resolve_path(str(resolved["local_model_dir"]), data_root)

        missing = [
            key
            for key, field in cls.model_fields.items()
            if field.is_required() and key not in resolved
        ]
        if missing:
            raise ValueError(f"The API configuration is incomplete; missing required keys: {', '.join(missing)}")

        return cls.model_validate(resolved)
