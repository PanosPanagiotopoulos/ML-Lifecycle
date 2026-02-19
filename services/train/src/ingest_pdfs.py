import os
from pathlib import Path
import pdfplumber
from logging_service import get_logger


LOGGER = get_logger("train.ingest")

def extract_text_from_pdf(pdf_path: str) -> str:
    parts: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text)
    result = "\n".join(parts)
    LOGGER.info("Extracted text from PDF: %s", os.path.basename(pdf_path))
    return result

def ingest_all(raw_pdfs_dir: str) -> list[dict]:
    LOGGER.info("Ingesting PDFs from: %s", raw_pdfs_dir)
    root = Path(raw_pdfs_dir)
    if not root.exists():
        LOGGER.error("Raw PDFs directory does not exist: %s", root)
        return []

    pdf_paths = [str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]
    if not pdf_paths:
        LOGGER.warning("No PDF files found under: %s", root.resolve())
        return []

    rows: list[dict] = []
    for path in pdf_paths:
        doc_id = os.path.basename(path)
        try:
            text = extract_text_from_pdf(path)
            if text.strip():
                rows.append({"doc_id": doc_id, "text": text})
            else:
                LOGGER.warning("Skipping PDF with no extracted text: %s", doc_id)
        except Exception as exc:
            LOGGER.exception("Failed to process PDF: %s | error=%s", doc_id, exc)
    LOGGER.info("PDF ingestion complete. Total documents: %s", len(rows))
    return rows
