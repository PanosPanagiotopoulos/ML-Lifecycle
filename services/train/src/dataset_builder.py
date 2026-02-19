import os
import json
from logging_service import get_logger


LOGGER = get_logger("train.dataset")

def build_jsonl(extracted_docs: list[dict], out_jsonl_path: str, chunk_size: int, overlap: int) -> int:
    from chunking import chunk_text

    LOGGER.info("Building JSONL dataset with chunk_size=%s, overlap=%s", chunk_size, overlap)
    os.makedirs(os.path.dirname(out_jsonl_path), exist_ok=True)
    chunks_jsonl_path = os.path.join(os.path.dirname(out_jsonl_path), "chunks.jsonl")

    n = 0
    with open(out_jsonl_path, "w", encoding="utf-8") as train_file, open(chunks_jsonl_path, "w", encoding="utf-8") as chunks_file:
        for doc in extracted_docs:
            doc_id = doc["doc_id"]
            chunks = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
            for i, ch in enumerate(chunks):
                train_sample = {
                    "text": ch,
                    "source": {"doc_id": doc_id, "chunk_index": i},
                }

                retrieval_chunk = {
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "text": ch,
                }

                train_file.write(json.dumps(train_sample, ensure_ascii=False) + "\n")
                chunks_file.write(json.dumps(retrieval_chunk, ensure_ascii=False) + "\n")
                n += 1

    LOGGER.info("JSONL dataset built. Total samples: %s", n)
    LOGGER.info("Retrieval chunks file generated at: %s", chunks_jsonl_path)
    return n
