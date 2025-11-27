"""PDF processing utilities for extracting text from school guides."""
import os
import logging
import warnings
from pathlib import Path
from typing import List, Dict
import re
from pypdf import PdfReader

from src.common.logger import get_logger
from src.common.config import Config

logger = get_logger("mllifecycle.pdf_processor")

# ---- Silence noisy pypdf warnings (best effort) ----
logging.getLogger("pypdf").setLevel(logging.ERROR)
# Some versions emit this as a warning, so we also filter by message:
warnings.filterwarnings(
    "ignore",
    message=r"Multiple definitions in dictionary.*"
)


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF using pypdf with robust error handling.

    - Uses strict=False to be tolerant of malformed PDFs.
    - On hard read errors, logs and returns empty string instead of raising.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return ""

    text = ""
    try:
        with open(pdf_path, "rb") as f:
            # strict=False makes parser more forgiving
            try:
                reader = PdfReader(f, strict=False)
            except Exception as e:
                # This is where the /Group dictionary issue typically occurs
                logger.warning(f"Skipping PDF {pdf_path.name} due to read error: {e}")
                return ""

            logger.info(f"Reading PDF {pdf_path.name} with {len(reader.pages)} pages")
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    logger.warning(
                        f"Failed to extract text from page {page_num} of {pdf_path.name}: {e}"
                    )
                    continue
    except Exception as e:
        logger.error(f"Failed to read PDF {pdf_path}: {e}")
        return ""

    return text


def chunk_text(
    text: str,
    chunk_size: int = None,
    overlap: int = None
) -> List[str]:
    chunk_size = chunk_size or Config.PDF_CHUNK_SIZE
    overlap = overlap or Config.PDF_CHUNK_OVERLAP

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap

    min_size = Config.PDF_MIN_CHUNK_SIZE
    return [c for c in chunks if len(c) > min_size]


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:\-\'"]', '', text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    return text.strip()


def extract_qa_pairs(text: str, qa_delimiter: str = "Q:") -> List[Dict[str, str]]:
    qa_pairs = []
    parts = text.split(qa_delimiter)

    for part in parts[1:]:
        if "A:" in part:
            q, a = part.split("A:", 1)
            qa_pairs.append({
                "question": q.strip(),
                "answer": a.strip()
            })

    return qa_pairs


def process_pdf_directory(pdf_dir: str, chunk_size: int = None) -> List[Dict[str, str]]:
    """Process all PDFs in a directory into chunked, cleaned documents.

    - Skips unreadable PDFs.
    - Skips PDFs with no extractable text.
    """
    chunk_size = chunk_size or Config.PDF_CHUNK_SIZE
    pdf_dir = Path(pdf_dir)

    # Convert relative path to absolute if needed
    if not pdf_dir.is_absolute():
        pdf_dir = Path.cwd() / pdf_dir

    logger.info(f"Looking for PDFs in: {pdf_dir}")

    if not pdf_dir.exists():
        logger.error(f"Directory does not exist: {pdf_dir}")
        return []

    documents = []
    pdf_files = list(pdf_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files")

    for pdf_file in pdf_files:
        logger.info(f"Processing {pdf_file.name}")
        text = extract_pdf_text(str(pdf_file))

        # If extraction failed or PDF is effectively empty, skip
        if not text or len(text.strip()) < 10:
            logger.warning(f"No meaningful text extracted from {pdf_file.name}, skipping")
            continue

        text = clean_text(text)
        chunks = chunk_text(text, chunk_size=chunk_size)

        for chunk in chunks:
            documents.append({
                "source": pdf_file.name,
                "text": chunk
            })

        logger.info(f"Extracted {len(chunks)} chunks from {pdf_file.name}")

    logger.info(f"Total documents extracted: {len(documents)}")
    return documents
