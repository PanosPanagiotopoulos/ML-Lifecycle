import os
from config import TrainConfig
from ingest_pdfs import ingest_all
from dataset_builder import build_jsonl
from train_lora import train
from logging_service import get_logger


LOGGER = get_logger("train.pipeline")

def main() -> None:
    LOGGER.info("ML Pipeline started")
    cfg = TrainConfig.load()
    LOGGER.info("Configuration loaded")
    
    docs = ingest_all(cfg.raw_pdfs_dir)
    if not docs:
        LOGGER.error("No PDFs/text extracted. Aborting before training.")
        return
    
    out_jsonl = os.path.join(cfg.processed_dir, "train.jsonl")
    n = build_jsonl(docs, out_jsonl, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap)
    LOGGER.info("Dataset build complete with %s samples", n)
    if n <= 0:
        LOGGER.error("Training dataset is empty (%s samples). Aborting before training.", n)
        return
    
    LOGGER.info("Training phase started")
    train()
    LOGGER.info("ML Pipeline complete")

if __name__ == "__main__":
    main()
