import os
import math
import inspect
import mlflow
import torch
from huggingface_hub import login, snapshot_download
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
from config import TrainConfig
from mlflow_registry import register_and_promote
from logging_service import get_logger


LOGGER = get_logger("train.lora")


def _count_non_empty_lines(file_path: str) -> int:
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _build_training_args(cfg: TrainConfig, non_empty_records: int) -> TrainingArguments:
    signature = inspect.signature(TrainingArguments.__init__)
    params = signature.parameters
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    kwargs: dict = {
        "output_dir": os.path.join(cfg.processed_dir, "artifacts_tmp"),
        "num_train_epochs": cfg.num_train_epochs,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "logging_steps": 25,
        "fp16": use_cuda and not use_bf16,
    }

    if "bf16" in params:
        kwargs["bf16"] = use_bf16

    if "gradient_checkpointing" in params:
        kwargs["gradient_checkpointing"] = use_cuda

    if "warmup_steps" in params:
        estimated_steps = max(
            1,
            int(
                ((1.0 - cfg.eval_ratio) * non_empty_records)
                / max(1, cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps)
            ) * max(1, cfg.num_train_epochs),
        )
        kwargs["warmup_steps"] = max(0, int(estimated_steps * cfg.warmup_ratio))
    elif "warmup_ratio" in params:
        kwargs["warmup_ratio"] = cfg.warmup_ratio

    if "dataloader_pin_memory" in params:
        kwargs["dataloader_pin_memory"] = torch.cuda.is_available()

    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"

    if "save_strategy" in params:
        kwargs["save_strategy"] = "epoch"

    if "report_to" in params:
        kwargs["report_to"] = []

    return TrainingArguments(**kwargs)


def _prepare_hf_runtime(cfg: TrainConfig) -> str:
    token = (getattr(cfg, "hf_token", None) or os.getenv("HF_TOKEN", "")).strip()
    if token == "HF_TOKEN":
        token = ""

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    cache_dir = (
        getattr(cfg, "hf_cache_dir", None)
        or os.getenv("HF_HOME")
        or os.getenv("HUGGINGFACE_HUB_CACHE")
        or None
    )

    if token:
        login(token=token, add_to_git_credential=False)

    return snapshot_download(
        repo_id=cfg.base_model_name,
        token=token or None,
        cache_dir=cache_dir,
        resume_download=True,
    )


def _load_base_model(model_source: str):
    use_cuda = torch.cuda.is_available()
    model_kwargs = {"low_cpu_mem_usage": True}
    if use_cuda:
        model_kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    try:
        return AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    except TypeError:
        if "dtype" in model_kwargs:
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        return AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)

def train() -> None:
    cfg = TrainConfig.load()
    LOGGER.info("Training config loaded")

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)
    LOGGER.info("MLflow initialized")

    set_seed(cfg.seed)
    LOGGER.info("Random seed set")

    LOGGER.info("Preparing Hugging Face runtime and local model snapshot")
    model_source = _prepare_hf_runtime(cfg)
    LOGGER.info("Base model snapshot prepared: %s", model_source)

    dataset_path = os.path.join(cfg.processed_dir, "train.jsonl")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")

    non_empty_records = _count_non_empty_lines(dataset_path)
    if non_empty_records == 0:
        raise ValueError(
            f"Training dataset is empty: {dataset_path}. "
            "Add PDFs to raw_pdfs and rebuild dataset before training."
        )

    ds = load_dataset("json", data_files=dataset_path, split="train")
    LOGGER.info("Dataset loaded (%s records)", len(ds))

    ds = ds.train_test_split(test_size=cfg.eval_ratio, seed=cfg.seed)
    LOGGER.info("Train/test split complete")

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    LOGGER.info("Tokenizer loaded")

    def tokenize(batch: dict) -> dict:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=cfg.max_length,
            padding="max_length",
        )

    tokenized = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)
    LOGGER.info("Dataset tokenized")

    base_model = _load_base_model(model_source)
    LOGGER.info("Base model loaded")

    use_fan_in_fan_out = getattr(base_model.config, "model_type", "") == "gpt2"

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        fan_in_fan_out=use_fan_in_fan_out,
    )
    model = get_peft_model(base_model, lora_config)
    LOGGER.info("LoRA model configured")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    LOGGER.info("Data collator ready")

    args = _build_training_args(cfg, non_empty_records=non_empty_records)

    with mlflow.start_run() as run:
        LOGGER.info("MLflow run started")
        mlflow.log_params({
            "base_model_name": cfg.base_model_name,
            "max_length": cfg.max_length,
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "lora_r": cfg.lora_r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "learning_rate": cfg.learning_rate,
            "epochs": cfg.num_train_epochs,
            "grad_accum": cfg.gradient_accumulation_steps,
        })
        LOGGER.info("Parameters logged to MLflow")

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["test"],
            data_collator=collator,
        )
        LOGGER.info("Trainer initialized")

        LOGGER.info("Model training started")
        trainer.train()
        LOGGER.info("Training complete")
        
        LOGGER.info("Model evaluation started")
        metrics = trainer.evaluate()
        LOGGER.info("Evaluation complete")

        eval_loss = float(metrics.get("eval_loss", 0.0))
        perplexity = math.exp(eval_loss) if eval_loss > 0 else 0.0
        LOGGER.info("Metrics calculated - Loss: %.4f, Perplexity: %.4f", eval_loss, perplexity)

        mlflow.log_metrics({"eval_loss": eval_loss, "perplexity": perplexity})
        LOGGER.info("Metrics logged to MLflow")

        out_dir = os.path.join(cfg.processed_dir, "model_out")
        os.makedirs(out_dir, exist_ok=True)

        LOGGER.info("Merging LoRA adapter into base model for deployable artifact")
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        LOGGER.info("Model and tokenizer saved")

        mlflow.log_artifacts(out_dir, artifact_path="model")
        LOGGER.info("Artifacts logged to MLflow")

        LOGGER.info("Model registration and promotion started")
        register_and_promote(run.info.run_id, artifact_path="model", model_name=cfg.registered_model_name)
        LOGGER.info("Model registered and promoted")

if __name__ == "__main__":
    train()
