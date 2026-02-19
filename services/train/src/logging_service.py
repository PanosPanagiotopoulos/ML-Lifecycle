import logging
import os


def get_logger(name: str, log_dir: str | None = None, log_file_name: str = "train_pipeline.log") -> logging.Logger:
    if log_dir is None:
        log_dir = os.getenv("LOG_DIR")
    if log_dir is None:
        base_dir = os.path.dirname(__file__)
        log_dir = os.path.abspath(os.path.join(base_dir, "../data/logs"))

    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_path = os.path.join(log_dir, log_file_name)
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger