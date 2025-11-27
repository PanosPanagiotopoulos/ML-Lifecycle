"""LLM fine-tuning for PDF-based question answering."""
import os
import sys
import logging
from pathlib import Path
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

from src.common.io_utils import load_yaml, ensure_dir
from src.common.logger import setup_logger, get_logger
from src.common.config import Config
from src.data.dataset import load_pdf_documents

logger = get_logger("mllifecycle.training")


class LLMTrainer:
    """Fine-tunes Hugging Face LLMs on PDF documents."""
    
    def __init__(self, config: dict):
        self.config = config
        self.model_dir = Path(config["trainer"].get("output_dir", str(Config.MODEL_DIR)))
        ensure_dir(str(self.model_dir))
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_data(self) -> tuple:
        """Load and prepare training data."""
        logger.info("STEP 0: Loading PDF Documents")
        
        pdf_dir = self.config["data"].get("pdf_dir", str(Config.RAW_DATA_DIR))
        chunk_size = self.config["data"].get("chunk_size", 512)
        test_split = self.config["data"].get("test_split", 0.2)
        seed = self.config.get("seed", 42)
        
        pdf_dir = Path(pdf_dir).resolve()
        documents = load_pdf_documents(str(pdf_dir), chunk_size=chunk_size)
        
        if not documents:
            raise ValueError(
                f"No PDF chunks found in {pdf_dir}. "
                f"Ensure you have at least one .pdf file in data/raw/"
            )
        
        # FOR TESTING: Limit to first 100 samples for faster training
        documents = documents[:100]
        
        logger.info(f"Loaded {len(documents)} chunks from {len(set(d['source'] for d in documents))} PDFs")
        
        import random
        random.seed(seed)
        random.shuffle(documents)
        
        n_train = int(len(documents) * (1 - test_split))
        train_docs = documents[:n_train]
        val_docs = documents[n_train:]
        
        logger.info(f"Train: {len(train_docs)}, Val: {len(val_docs)}")
        
        return train_docs, val_docs
    
    def prepare_dataset(self, documents: list) -> Dataset:
        """Convert documents to instruction-following format."""
        formatted_data = []
        
        for doc in documents:
            text = doc["text"]
            source = doc["source"]
            
            instruction = (
                f"Read the following study guide content from '{source}'. "
                "You will later be asked questions about it."
            )
            prompt = f"### Instruction:\n{instruction}\n\n### Content:\n{text}\n\n### Notes:\n"
            
            formatted_data.append({
                "text": prompt,
                "source": source
            })
        
        dataset = Dataset.from_list(formatted_data)
        logger.info(f"Dataset prepared: {len(dataset)} samples")
        
        return dataset
    
    def load_model_and_tokenizer(self):
        """Load pre-trained LLM and tokenizer from Hugging Face."""
        logger.info("STEP 1: Loading Pre-trained LLM")
        
        model_name = self.config["trainer"].get("model_name", "gpt2")
        use_4bit = self.config["trainer"].get("use_4bit", True)
        
        if self.device == "cpu" and model_name in ["microsoft/phi-2", "mistralai/Mistral-7B-Instruct-v0.2"]:
            logger.warning(
                f"Running {model_name} on CPU will be slow. "
                "Consider using a GPU or a smaller model like 'gpt2'."
            )
        
        logger.info(f"Model: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side="right"
            )
        except Exception as e:
            raise ValueError(f"Cannot load tokenizer for '{model_name}'. Check model name or internet connection.") from e
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        bnb_config = None
        if use_4bit and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            logger.info("4-bit quantization enabled")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config if use_4bit else None,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
        except Exception as e:
            if "out of memory" in str(e).lower():
                raise MemoryError(
                    f"Out of memory loading {model_name}. "
                    "Try: use_4bit=true, smaller model, or reduce batch size."
                ) from e
            raise
        
        total_params = self.model.num_parameters()
        logger.info(f"Model loaded ({total_params / 1e9:.2f}B params)")
        
        use_lora = self.config["trainer"].get("use_lora", True)
        if use_lora:
            if use_4bit:
                self.model = prepare_model_for_kbit_training(self.model)
            
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            lora_config = LoraConfig(
                r=self.config["trainer"].get("lora_r", 16),
                lora_alpha=self.config["trainer"].get("lora_alpha", 32),
                target_modules=target_modules,
                lora_dropout=self.config["trainer"].get("lora_dropout", 0.05),
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            self.model = get_peft_model(self.model, lora_config)
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(f"LoRA: {trainable/1e6:.1f}M trainable ({100*trainable/total_params:.2f}%)")
    
    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize the dataset."""
        max_length = self.config["trainer"].get("max_length", 512)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train_model(self, train_dataset: Dataset, eval_dataset: Dataset):
        """Fine-tune the LLM."""
        logger.info("STEP 2: Fine-tuning LLM")
        
        num_epochs = self.config["trainer"].get("num_train_epochs", 3)
        batch_size = self.config["trainer"].get("per_device_train_batch_size", 2)
        fp16_enabled = self.config["trainer"].get("fp16", False) and self.device == "cuda"
        bf16_enabled = self.config["trainer"].get("bf16", False) and self.device == "cuda"
        
        logger.info(f"Epochs: {num_epochs}, Batch size: {batch_size}, Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
        
        training_args = TrainingArguments(
            output_dir=str(self.model_dir / "checkpoints"),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=self.config["trainer"].get("per_device_eval_batch_size", 2),
            gradient_accumulation_steps=self.config["trainer"].get("gradient_accumulation_steps", 4),
            learning_rate=float(self.config["trainer"].get("learning_rate", 2e-4)),
            warmup_steps=self.config["trainer"].get("warmup_steps", 100),
            max_grad_norm=float(self.config["trainer"].get("max_grad_norm", 0.3)),
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            eval_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=True,
            fp16=fp16_enabled,
            bf16=bf16_enabled,
            report_to="none",
            remove_unused_columns=False,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        try:
            self.trainer.train()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(
                    "CUDA out of memory. Try reducing per_device_train_batch_size "
                    "or max_length in configs/train.yaml"
                )
            raise
    
    def evaluate_model(self) -> dict:
        """Evaluate the fine-tuned model."""
        logger.info("STEP 3: Evaluation")
        
        metrics = self.trainer.evaluate()
        eval_loss = metrics.get('eval_loss', None)
        
        if eval_loss is not None:
            perplexity = torch.exp(torch.tensor(eval_loss))
            logger.info(f"Eval loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")
        
        return metrics
    
    def save_model(self):
        """Save the fine-tuned model and configuration for API serving."""
        logger.info("STEP 4: Saving Model")
        
        self.model.save_pretrained(str(self.model_dir))
        self.tokenizer.save_pretrained(str(self.model_dir))
        
        config_dict = {
            "model_name": self.config["trainer"].get("model_name", "gpt2"),
            "use_lora": self.config["trainer"].get("use_lora", True),
            "lora_r": self.config["trainer"].get("lora_r", 16),
            "max_length": self.config["trainer"].get("max_length", 512),
            "temperature": self.config["trainer"].get("temperature", 0.7),
            "top_p": self.config["trainer"].get("top_p", 0.9),
        }
        
        config_path = self.model_dir / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model saved to {self.model_dir}")
    
    def train(self) -> dict:
        """Run complete LLM fine-tuning pipeline."""
        logger.info("Starting LLM Fine-tuning Pipeline")
        
        try:
            train_docs, val_docs = self.load_data()
            train_dataset = self.prepare_dataset(train_docs)
            val_dataset = self.prepare_dataset(val_docs)
            
            self.load_model_and_tokenizer()
            
            train_dataset = self.tokenize_dataset(train_dataset)
            val_dataset = self.tokenize_dataset(val_dataset)
            
            self.train_model(train_dataset, val_dataset)
            metrics = self.evaluate_model()
            self.save_model()
            
            logger.info("Fine-tuning complete")
            
            return {"metrics": metrics}
            
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point for training pipeline."""
    if len(sys.argv) < 2:
        print("Usage: python train.py <config_path>")
        print("Example: python train.py configs/train.yaml")
        sys.exit(1)
    
    config_path = sys.argv[1]
    
    try:
        cfg = load_yaml(config_path)
    except Exception as e:
        print(f"Error loading config from {config_path}: {e}")
        sys.exit(1)
    
    Config.ensure_directories()
    
    log_level = cfg.get("log_level", "INFO")
    log_file = "logs/training.log"
    setup_logger(name="mllifecycle", level=getattr(logging, log_level), log_file=log_file)
    
    main_logger = get_logger("mllifecycle")
    main_logger.info("Starting LLM Fine-tuning Pipeline")
    main_logger.info(f"Config: {config_path}")
    
    seed = cfg.get("seed", 42)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    try:
        trainer = LLMTrainer(cfg)
        results = trainer.train()
        
        final_loss = results['metrics'].get('eval_loss', 0)
        main_logger.info(f"Final eval loss: {final_loss:.4f}")
        main_logger.info(f"Model saved to: {trainer.model_dir}")
        
    except ValueError as e:
        main_logger.error(f"Configuration or data error: {str(e)}")
        sys.exit(1)
    except MemoryError as e:
        main_logger.error(f"Memory error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        main_logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
