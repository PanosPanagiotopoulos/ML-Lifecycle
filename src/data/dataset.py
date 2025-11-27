"""Dataset loading for PDF-based training."""
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from datasets import Dataset
import numpy as np

from src.data.pdf_processor import process_pdf_directory
from src.common.logger import get_logger
from src.common.config import Config

logger = get_logger("mllifecycle.dataset")


def load_pdf_documents(pdf_dir: str, chunk_size: int = None) -> List[Dict[str, str]]:
    chunk_size = chunk_size or Config.PDF_CHUNK_SIZE
    logger.info(f"Loading PDFs from {pdf_dir}")
    documents = process_pdf_directory(pdf_dir, chunk_size=chunk_size)
    logger.info(f"Loaded {len(documents)} document chunks")
    return documents


def create_training_dataset(
    documents: List[Dict[str, str]],
    labels: Optional[List[int]] = None,
    test_split: float = None
) -> Tuple[Dataset, Dataset]:
    test_split = test_split or Config.TEST_SPLIT
    
    if not documents:
        raise ValueError("No documents provided")
    
    df = pd.DataFrame(documents)
    
    if labels is not None:
        if len(labels) != len(df):
            raise ValueError(f"Labels length ({len(labels)}) != documents length ({len(df)})")
        df["label"] = labels
    else:
        unique_sources = df["source"].unique()
        source_to_label = {src: i for i, src in enumerate(unique_sources)}
        df["label"] = df["source"].map(source_to_label)
    
    n_train = int(len(df) * (1 - test_split))
    train_df = df[:n_train]
    val_df = df[n_train:]
    
    train_dataset = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False)
    val_dataset = Dataset.from_pandas(val_df[["text", "label"]], preserve_index=False)
    
    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    return train_dataset, val_dataset


def load_from_csv(csv_path: str, text_col: str = "text", label_col: str = "label") -> Dataset:
    df = pd.read_csv(csv_path)
    if text_col not in df or label_col not in df:
        raise ValueError(f"Missing required columns: {text_col}, {label_col}")
    
    return Dataset.from_pandas(df[[text_col, label_col]], preserve_index=False)

