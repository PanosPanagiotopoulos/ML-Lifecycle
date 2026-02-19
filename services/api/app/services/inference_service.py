import time
import json
import os
import re
import logging
import mlflow
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import ApiConfig

PROMPT_SUFFIX = "\n\nAnswer:"


LOGGER = logging.getLogger("api.inference")

class InferenceService:
    def __init__(self, cfg: ApiConfig) -> None:
        self._cfg = cfg
        self._retrieval_top_k = max(1, cfg.retrieval_top_k)
        LOGGER.info("Using MLflow tracking URI: %s", cfg.mlflow_tracking_uri)
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        self._client = mlflow.tracking.MlflowClient()
        self._model_cache: dict[str, tuple[AutoTokenizer, AutoModelForCausalLM]] = {}
        self._models_cache_ttl_seconds = 10.0
        self._models_cache_at = 0.0
        self._models_cache: list[dict] = []

        self._model_cache[cfg.model_name] = self._load_model(cfg.model_name)

        self._chunks = self._load_chunks(cfg.processed_chunks_path)

    def _load_model(self, model_name: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        uri = f"models:/{model_name}/{self._cfg.model_stage}"
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
        now = time.time()
        if self._models_cache and now - self._models_cache_at <= self._models_cache_ttl_seconds:
            return self._models_cache

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

    def chunk_count(self) -> int:
        return len(self._chunks)

    def _load_model_and_tokenizer(self, local_dir: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        adapter_config_path = os.path.join(local_dir, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            LOGGER.info("Detected adapter artifact. Loading base model + LoRA adapter.")
            peft_config = PeftConfig.from_pretrained(local_dir)
            tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
            base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
            peft_model = PeftModel.from_pretrained(base_model, local_dir)
            model = peft_model.merge_and_unload()
            return tokenizer, model

        LOGGER.info("Detected merged model artifact. Loading directly from registry artifact.")
        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        model = AutoModelForCausalLM.from_pretrained(local_dir)
        return tokenizer, model

    def _load_chunks(self, chunks_path: str) -> list[str]:
        if not os.path.exists(chunks_path):
            LOGGER.warning("Chunks file not found at %s. Retrieval context disabled.", chunks_path)
            return []

        rows: list[str] = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                text = (row.get("text") or "").strip()
                if text:
                    rows.append(text)

        LOGGER.info("Loaded retrieval chunks: %s", len(rows))
        return rows

    def _tokenize_for_retrieval(self, text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))

    def _retrieve_context(self, question: str) -> list[str]:
        if not self._chunks:
            return []

        q_tokens = self._tokenize_for_retrieval(question)
        if not q_tokens:
            return []

        scored: list[tuple[float, str]] = []
        for chunk in self._chunks:
            c_tokens = self._tokenize_for_retrieval(chunk)
            if not c_tokens:
                continue
            overlap = q_tokens.intersection(c_tokens)
            if not overlap:
                continue
            score = len(overlap) / (len(q_tokens) ** 0.5 * len(c_tokens) ** 0.5)
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[: self._retrieval_top_k]]

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

        retrieved_chunks = self._retrieve_context(question)
        context_blocks: list[str] = []
        if course_guide_context and course_guide_context.strip():
            context_blocks.append(course_guide_context.strip())
        context_blocks.extend(retrieved_chunks)

        context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "No additional course-guide context available."
        prompt = (
            "You are a university course-guide assistant. Use the provided context when relevant. "
            "If context is insufficient, answer cautiously and say what is missing.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}{PROMPT_SUFFIX}"
        )

        start = time.perf_counter()

        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        answer = text.split(PROMPT_SUFFIX, 1)[-1].strip()

        latency_ms = (time.perf_counter() - start) * 1000.0
        return answer, latency_ms
