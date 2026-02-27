import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.routes.openai import router as openai_router
from app.routes.health import router as health_router

app = FastAPI(title="UniGuide LLM API")

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

LOGGER = logging.getLogger("api.error")


def _error_payload(
	status_code: int,
	code: str,
	message: str,
	path: str,
	details: Any = None,
) -> dict[str, Any]:
	payload: dict[str, Any] = {
		"error": {
			"code": code,
			"message": message,
			"status": status_code,
			"path": path,
		}
	}
	if details is not None:
		payload["error"]["details"] = details
	return payload


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
	details = [
		{
			"field": ".".join(str(part) for part in err.get("loc", [])),
			"message": err.get("msg", "Invalid value."),
			"type": err.get("type", "validation_error"),
		}
		for err in exc.errors()
	]
	return JSONResponse(
		status_code=422,
		content=_error_payload(
			status_code=422,
			code="validation_error",
			message="The request payload is invalid.",
			path=request.url.path,
			details=details,
		),
	)


@app.exception_handler(StarletteHTTPException)
async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
	default_messages = {
		400: "The request is invalid.",
		401: "Authentication is required to access this resource.",
		403: "You are not authorized to access this resource.",
		404: f"No endpoint matches {request.method} {request.url.path}; verify the URL and HTTP method.",
		405: f"The method {request.method} is not allowed for {request.url.path}; use a supported method for this endpoint.",
		409: "The request conflicts with the current state of the resource.",
		415: "The request media type is not supported.",
		429: "Too many requests were sent; please retry later.",
		500: "An internal server error occurred.",
		503: "The service is temporarily unavailable.",
	}
	framework_details = {"Not Found", "Method Not Allowed", "Bad Request", "Unauthorized", "Forbidden"}
	if isinstance(exc.detail, str) and exc.detail.strip() and exc.detail.strip() not in framework_details:
		message = exc.detail.strip()
	else:
		message = default_messages.get(exc.status_code, "The request could not be processed.")
	code = f"http_{exc.status_code}"
	return JSONResponse(
		status_code=exc.status_code,
		content=_error_payload(
			status_code=exc.status_code,
			code=code,
			message=message,
			path=request.url.path,
		),
	)


@app.exception_handler(Exception)
async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
	LOGGER.exception("Unhandled API exception: %s", exc)
	return JSONResponse(
		status_code=500,
		content=_error_payload(
			status_code=500,
			code="internal_error",
			message="An unexpected error occurred while processing the request.",
			path=request.url.path,
		),
	)

app.include_router(health_router)
app.include_router(openai_router)
