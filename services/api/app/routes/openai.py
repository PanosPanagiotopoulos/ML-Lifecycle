import time
import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from deps import get_config, get_inference_service
from config import ApiConfig
from services.inference_service import InferenceService


router = APIRouter(prefix="/v1")


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1, max_length=8000)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = None
    temperature: float | None = None


@router.get("/models")
def list_models(svc: InferenceService = Depends(get_inference_service)) -> dict:
    models = svc.list_models()
    return {
        "object": "list",
        "data": models,
    }


@router.post("/chat/completions")
def create_chat_completion(
    req: ChatCompletionRequest,
    cfg: ApiConfig = Depends(get_config),
    svc: InferenceService = Depends(get_inference_service),
) -> dict:
    model_ids = {m["id"] for m in svc.list_models()}
    if req.model not in model_ids:
        raise HTTPException(status_code=404, detail="model_not_available")

    user_messages = [m.content for m in req.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="At least one user message is required")

    system_messages = [m.content for m in req.messages if m.role == "system"]
    system_context = "\n\n".join(system_messages).strip() if system_messages else None
    question = user_messages[-1]

    try:
        max_tokens = req.max_tokens if req.max_tokens is not None else cfg.chat_default_max_tokens
        temperature = req.temperature if req.temperature is not None else cfg.chat_default_temperature
        answer, latency_ms = svc.ask(
            question=question,
            max_new_tokens=max_tokens,
            temperature=temperature,
            course_guide_context=system_context,
            model_name=req.model,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail="inference_unavailable") from exc

    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    prompt_tokens_est = max(1, len(question.split()))
    completion_tokens_est = max(1, len(answer.split()))

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens_est,
            "completion_tokens": completion_tokens_est,
            "total_tokens": prompt_tokens_est + completion_tokens_est,
        },
        "latency_ms": latency_ms,
    }
