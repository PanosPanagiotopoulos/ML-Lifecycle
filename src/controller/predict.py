"""FastAPI application for LLM-based question answering."""
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.common.logger import setup_logger, get_logger
from src.common.config import Config

logger = get_logger(__name__)

Config.ensure_directories()

model = None
tokenizer = None
generation_config = {}

def load_llm_model():
    """Load the fine-tuned LLM model and tokenizer."""
    global model, tokenizer, generation_config
    
    try:
        model_path = Config.MODEL_DIR
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model directory not found: {model_path}. "
                "Train a model first using: python train.py configs/train.yaml"
            )
        
        logger.info(f"Loading model from {model_path}")
        
        config_path = model_path / "training_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                generation_config = {
                    "max_length": config.get("max_length", 512),
                    "temperature": config.get("temperature", 0.7),
                    "top_p": config.get("top_p", 0.9),
                }
                base_model_name = config.get("model_name", "gpt2")
        else:
            logger.warning(f"training_config.json not found, using defaults")
            base_model_name = "gpt2"
            generation_config = {"max_length": 512, "temperature": 0.7, "top_p": 0.9}
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True
            )
        except Exception as e:
            raise ValueError(f"Cannot load tokenizer. Model files may be corrupted.") from e
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("Tokenizer loaded")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        adapter_config_path = model_path / "adapter_config.json"
        
        if adapter_config_path.exists():
            logger.info("Loading base model + LoRA adapter")
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
                model = PeftModel.from_pretrained(base_model, str(model_path))
            except Exception as e:
                raise ValueError(
                    f"Cannot load LoRA model. Ensure base model '{base_model_name}' is accessible."
                ) from e
        else:
            logger.info("Loading full fine-tuned model")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
            except Exception as e:
                raise ValueError("Cannot load fine-tuned model. Model files may be corrupted.") from e
        
        model.eval()
        logger.info(f"Model loaded on {device}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise

try:
    load_llm_model()
except Exception as e:
    logger.error(f"Model loading failed: {e}")

app = FastAPI(
    title="PDF LLM API",
    version="2.0.0",
    description="Question answering API using fine-tuned LLM on PDF documents"
)


class QuestionInput(BaseModel):
    question: str
    max_length: int = 200
    temperature: float = 0.7
    top_p: float = 0.9


class AnswerOutput(BaseModel):
    answer: str
    question: str


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.post("/ask", response_model=AnswerOutput)
async def ask_question(inp: QuestionInput):
    """Ask a question about the PDF documents the model was trained on."""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Format prompt to match training format
        instruction = (
            "Read the following study guide content. "
            "You will later be asked questions about it."
        )
        prompt = f"### Instruction:\n{instruction}\n\n### Content:\n{inp.question}\n\n### Notes:\n"
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inp.max_length or generation_config.get("max_length", 200),
                temperature=inp.temperature or generation_config.get("temperature", 0.7),
                top_p=inp.top_p or generation_config.get("top_p", 0.9),
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer after the prompt
        if "### Notes:" in full_response:
            answer = full_response.split("### Notes:")[-1].strip()
        else:
            answer = full_response.replace(prompt, "").strip()
        
        logger.info(f"Q: {inp.question[:50]}... | A: {answer[:50]}...")
        
        return {
            "question": inp.question,
            "answer": answer
        }
        
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
