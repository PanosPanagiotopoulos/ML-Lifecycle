import logging
from fastapi import FastAPI

from routes.openai import router as openai_router
from routes.health import router as health_router

app = FastAPI(title="UniGuide LLM API")

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app.include_router(health_router)
app.include_router(openai_router)
