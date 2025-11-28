"""Production FastAPI application for LLM inference."""
import time
from pathlib import Path
from typing import Optional
import json

import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.common.logger import get_logger
from src.common.config import Config
from src.dtos.requests import InferenceRequest, HealthCheckResponse, InferenceResponse
from src.monitoring.request_logger import RequestLogger

logger = get_logger(__name__)
Config.ensure_directories()

model: Optional[AutoModelForCausalLM] = None
tokenizer: Optional[AutoTokenizer] = None
generation_config: dict = {}
request_logger = RequestLogger()


def load_model() -> bool:
    """Load fine-tuned model with LoRA adapter."""
    global model, tokenizer, generation_config
    
    try:
        model_path = Config.MODEL_DIR
        
        if not model_path.exists():
            logger.warning(f"Model not found at {model_path}. Run training first.")
            return False
        
        logger.info(f"Loading model from {model_path}")
        
        config_path = model_path / "training_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                generation_config = {
                    "max_length": config.get("max_length", 128),
                    "temperature": config.get("temperature", 0.7),
                    "top_p": config.get("top_p", 0.9),
                }
                base_model_name = config.get("model_name", "distilgpt2")
        else:
            base_model_name = "distilgpt2"
            generation_config = {"max_length": 128, "temperature": 0.7, "top_p": 0.9}
        
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        adapter_config_path = model_path / "adapter_config.json"
        
        if adapter_config_path.exists():
            logger.info(f"Loading base model '{base_model_name}' with LoRA adapter")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            model = PeftModel.from_pretrained(base_model, str(model_path))
        else:
            logger.info("Loading full fine-tuned model")
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
        
        model.eval()
        logger.info(f"Model loaded successfully on {device}")
        return True
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}", exc_info=True)
        return False


app = FastAPI(
    title="Document QA API",
    version="1.0.0",
    description="Enterprise LLM inference service with monitoring",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    logger.info("Starting inference service...")
    success = load_model()
    if not success:
        logger.warning("Service started without model. Training required.")


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Service health status."""
    return HealthCheckResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_path=str(Config.MODEL_DIR)
    )


@app.get("/metrics")
async def get_metrics(date: Optional[str] = None):
    """Get inference metrics for specified date."""
    return request_logger.get_stats(date)


@app.post("/ask", response_model=InferenceResponse, status_code=status.HTTP_200_OK)
async def generate_answer(request: InferenceRequest):
    """
    Generate answer for given question.
    
    Args:
        request: Question and generation parameters
        
    Returns:
        Generated answer with metadata
        
    Raises:
        HTTPException: If model unavailable or generation fails
    """
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not available. Run training pipeline first."
        )
    
    start_time = time.time()
    
    try:
        prompt = f"Question: {request.question}\n\nAnswer:"
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=generation_config.get("max_length", 128)
        )
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response.replace(prompt, "").strip()
        
        if not answer:
            answer = "Unable to generate response."
        
        latency_ms = (time.time() - start_time) * 1000
        request_logger.log(
            question=request.question,
            answer=answer,
            latency_ms=latency_ms,
            metadata={
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }
        )
        
        logger.info(f"Question processed | Latency: {latency_ms:.0f}ms")
        
        return InferenceResponse(question=request.question, answer=answer)
        
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
