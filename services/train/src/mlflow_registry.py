import mlflow
import time
from logging_service import get_logger


LOGGER = get_logger("train.registry")


def _ensure_registered_model_exists(client: mlflow.tracking.MlflowClient, model_name: str) -> None:
    try:
        client.create_registered_model(model_name)
        LOGGER.info("Created registered model: %s", model_name)
    except Exception as exc:
        message = str(exc).lower()
        if "already exists" in message or "resource_already_exists" in message:
            LOGGER.info("Registered model already exists: %s", model_name)
            return
        raise


def _wait_until_ready(
    client: mlflow.tracking.MlflowClient,
    model_name: str,
    version: str,
    timeout_seconds: int = 120,
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        mv = client.get_model_version(name=model_name, version=version)
        status = str(getattr(mv, "status", ""))
        if status.upper() == "READY":
            return
        if status.upper() in {"FAILED_REGISTRATION", "FAILED"}:
            raise RuntimeError(f"Model version registration failed with status={status}")
        time.sleep(2)
    raise TimeoutError(f"Model version {model_name}/{version} did not become READY in time")


def register_and_promote(run_id: str, artifact_path: str, model_name: str) -> None:
    client = mlflow.tracking.MlflowClient()
    _ensure_registered_model_exists(client, model_name)

    model_uri = f"runs:/{run_id}/{artifact_path}"
    LOGGER.info("Registering model from URI: %s", model_uri)

    try:
        result = mlflow.register_model(model_uri=model_uri, name=model_name)
    except Exception as exc:
        LOGGER.warning(
            "register_model failed for URI '%s'. Falling back to create_model_version from run artifacts. error=%s",
            model_uri,
            exc,
        )
        run = client.get_run(run_id)
        artifact_root = run.info.artifact_uri.rstrip("/")
        source_uri = f"{artifact_root}/{artifact_path.strip('/')}"
        LOGGER.info("Fallback create_model_version source: %s", source_uri)
        result = client.create_model_version(name=model_name, source=source_uri, run_id=run_id)

    LOGGER.info("Model registered: name=%s version=%s", model_name, result.version)

    _wait_until_ready(client, model_name, str(result.version))

    LOGGER.info("Promoting model version %s to Production", result.version)
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Production",
        archive_existing_versions=True
    )
    LOGGER.info("Model promotion complete")
