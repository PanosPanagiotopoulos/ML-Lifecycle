"""API request and response models."""
from pydantic import BaseModel, Field, field_validator


class InferenceRequest(BaseModel):
    """Request model for text generation."""
    question: str = Field(..., min_length=1, max_length=1000, description="Input question")
    max_tokens: int = Field(100, ge=10, le=500, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.1, le=2.0, description="Sampling temperature")
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()


class InferenceResponse(BaseModel):
    """Response model for text generation."""
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Model availability")
    device: str = Field(..., description="Compute device")
    model_path: str = Field(..., description="Model artifact path")
