FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY config ./config
RUN mkdir -p /app/data/raw_pdfs /app/data/processed /app/data/logs

ENV TRAIN_CONFIG_JSON=/app/config/settings.json

CMD ["python", "src/run_pipeline.py"]
