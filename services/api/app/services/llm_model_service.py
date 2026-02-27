import time
import os
import re
import logging
import mlflow
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.config import ApiConfig

PROMPT_SUFFIX = "\n\nAnswer:"


LOGGER = logging.getLogger("api.inference")

class LLMModelService:
    def __init__(self, cfg: ApiConfig) -> None:
        self._cfg = cfg
        self._source_mode = cfg.model_source
        self._client = None
        if self._source_mode == "registry":
            LOGGER.info("Using MLflow tracking URI: %s", cfg.mlflow_tracking_uri)
            mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
            self._client = mlflow.tracking.MlflowClient()
        else:
            LOGGER.info("Using local model source: %s", cfg.local_model_dir)

        self._model_cache: dict[str, tuple[AutoTokenizer, AutoModelForCausalLM]] = {}
        self._models_cache_ttl_seconds = 10.0
        self._models_cache_at = 0.0
        self._models_cache: list[dict] = []

        self._model_cache[cfg.model_name] = self._load_model(cfg.model_name)

    def _artifact_uri_for_model(self, model_name: str) -> str:
        if self._cfg.registry_model_uri:
            uri = self._cfg.registry_model_uri
            uri = uri.replace("{model}", model_name)
            uri = uri.replace("{stage}", self._cfg.model_stage)
            return uri
        return f"models:/{model_name}/{self._cfg.model_stage}"

    def _load_model(self, model_name: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        if self._source_mode == "local":
            if not self._cfg.local_model_dir:
                raise ValueError("LOCAL_MODEL_DIR must be configured when MODEL_SOURCE is set to 'local'.")
            local_dir = self._cfg.local_model_dir
            if model_name and os.path.basename(local_dir) != model_name:
                candidate = os.path.join(local_dir, model_name)
                if os.path.exists(candidate):
                    local_dir = candidate
            if not os.path.exists(local_dir):
                raise FileNotFoundError(
                    f"The configured LOCAL_MODEL_DIR does not exist or is not accessible: {local_dir}."
                )
        else:
            uri = self._artifact_uri_for_model(model_name)
            local_dir = mlflow.artifacts.download_artifacts(artifact_uri=uri)

        tokenizer, model = self._load_model_and_tokenizer(local_dir)

        model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer, model

    def _get_or_load_model(self, model_name: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        cached = self._model_cache.get(model_name)
        if cached is not None:
            return cached

        loaded = self._load_model(model_name)
        self._model_cache[model_name] = loaded
        return loaded

    def list_models(self) -> list[dict]:
        if self._source_mode == "local":
            return [
                {
                    "id": self._cfg.model_name,
                    "object": "model",
                    "owned_by": "local-filesystem",
                    "version": "local",
                    "stage": self._cfg.model_stage,
                }
            ]

        now = time.time()
        if self._models_cache and now - self._models_cache_at <= self._models_cache_ttl_seconds:
            return self._models_cache

        if self._client is None:
            raise RuntimeError(
                "MLflow client is not initialized while MODEL_SOURCE is set to 'registry'; verify tracking configuration."
            )

        models = self._client.search_registered_models()
        rows: list[dict] = []

        for model in models:
            latest = getattr(model, "latest_versions", []) or []
            version_for_stage = None
            for version in latest:
                if str(getattr(version, "current_stage", "")) == self._cfg.model_stage:
                    version_for_stage = version
                    break

            if version_for_stage is None:
                continue

            rows.append(
                {
                    "id": model.name,
                    "object": "model",
                    "owned_by": "mlflow-registry",
                    "version": str(getattr(version_for_stage, "version", "")),
                    "stage": self._cfg.model_stage,
                }
            )

        rows.sort(key=lambda x: x["id"])
        self._models_cache = rows
        self._models_cache_at = now
        return rows

    def model_is_ready(self) -> bool:
        try:
            tokenizer, model = self._get_or_load_model(self._cfg.model_name)
            return tokenizer is not None and model is not None
        except Exception:
            return False

    def _load_model_and_tokenizer(self, local_dir: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        adapter_config_path = os.path.join(local_dir, "adapter_config.json")
        hf_token = os.getenv("HF_TOKEN", "").strip() or None
        if hf_token == "HF_TOKEN":
            hf_token = None

        if os.path.exists(adapter_config_path):
            LOGGER.info("Detected adapter artifact. Loading base model + LoRA adapter.")
            peft_config = PeftConfig.from_pretrained(local_dir)
            tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, token=hf_token)
            base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, token=hf_token)
            peft_model = PeftModel.from_pretrained(base_model, local_dir)
            model = peft_model.merge_and_unload()
            return tokenizer, model

        LOGGER.info("Detected merged model artifact. Loading directly from registry artifact.")
        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        model = AutoModelForCausalLM.from_pretrained(local_dir)
        return tokenizer, model

    def ask(
        self,
        question: str,
        max_new_tokens: int,
        temperature: float,
        course_guide_context: str | None = None,
        model_name: str | None = None,
    ) -> tuple[str, float]:
        selected_model = model_name or self._cfg.model_name
        tokenizer, model = self._get_or_load_model(selected_model)

        if course_guide_context and course_guide_context.strip():
            context_text = course_guide_context.strip()
            prompt = (
                "You are a university course-guide assistant. Use the provided context when relevant. "
                "If context is insufficient, answer cautiously and say what is missing.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {question}{PROMPT_SUFFIX}"
            )
        else:
            prompt = (
                "You are a university course-guide assistant. Answer the question directly and concisely.\n\n"
                f"Question: {question}{PROMPT_SUFFIX}"
            )

        start = time.perf_counter()

        inputs = tokenizer(prompt, return_tensors="pt")
        capped_max_new_tokens = max(16, min(max_new_tokens, 160))
        use_sampling = temperature > 0.25

        generation_kwargs = {
            **inputs,
            "max_new_tokens": capped_max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 3,
            "do_sample": use_sampling,
        }
        if use_sampling:
            generation_kwargs["temperature"] = max(0.2, min(temperature, 0.9))
            generation_kwargs["top_p"] = 0.9

        with torch.no_grad():
            output = model.generate(**generation_kwargs)

        text = tokenizer.decode(output[0], skip_special_tokens=True)

        answer = text.split(PROMPT_SUFFIX, 1)[-1].strip()
        answer = re.split(r"\n+\s*Question\s*:\s*", answer, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        while answer.lower().startswith("answer:"):
            answer = answer[len("answer:"):].strip()
        if not answer:
            answer = "I could not generate a reliable answer for this request."

        latency_ms = (time.perf_counter() - start) * 1000.0
        return answer, latency_ms
