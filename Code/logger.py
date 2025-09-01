"""
logger.py — єдиний фабричний логер для скриптів пайплайна.

Приклад:
    from logger import setup_logger
    logger = setup_logger("./runs/train/train.log",
                          logger_name="train", level="INFO",
                          capture_warnings=True, use_utc=True)
    logger.info("hello")

Особливості:
- Консоль + (опційно) файловий хендлер із ротацією за розміром.
- Без дублювання хендлерів; оновлює рівень/формат при повторних викликах.
- Опції: UTC-час, force-перезбір, capture_warnings.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Union

_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR":    logging.ERROR,
    "WARNING":  logging.WARNING,
    "INFO":     logging.INFO,
    "DEBUG":    logging.DEBUG,
    "NOTSET":   logging.NOTSET,
}

def _as_level(level: Union[str, int]) -> int:
    if isinstance(level, int):
        return level
    return _LEVELS.get(str(level).upper(), logging.INFO)

class _Formatter(logging.Formatter):
    """Форматер з підтримкою UTC-часу."""
    def __init__(self, fmt: str, datefmt: Optional[str], use_utc: bool):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.converter = time.gmtime if use_utc else time.localtime  # type: ignore[attr-defined]

def _handler_is_console(h: logging.Handler) -> bool:
    return isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout

def _handler_is_file(h: logging.Handler, file_path: Path) -> bool:
    return hasattr(h, "baseFilename") and Path(getattr(h, "baseFilename")).resolve() == file_path.resolve()

def setup_logger(
    log_file: Optional[Union[str, Path]] = None,
    *,
    logger_name: str = "qna",
    level: Union[str, int] = "INFO",
    propagate: bool = False,
    # Ротація за розміром:
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    # Формат/час:
    use_utc: bool = False,
    datefmt: str = "%Y-%m-%dT%H:%M:%S%z",
    # Поведінка при повторному виклику:
    force: bool = False,           # якщо True — прибирає всі хендлери і збирає наново
    update_existing: bool = True,  # якщо True — оновлює рівень/формат існуючих хендлерів
    # Інше:
    capture_warnings: bool = False,
) -> logging.Logger:
    """
    Створює/повертає налаштований логер без дублю хендлерів.
    """
    logger = logging.getLogger(logger_name)
    desired_level = _as_level(level)
    logger.setLevel(desired_level)
    logger.propagate = propagate

    if capture_warnings:
        logging.captureWarnings(True)

    # Форматер
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    formatter = _Formatter(fmt=fmt, datefmt=datefmt, use_utc=use_utc)

    # Повністю перебудувати логер
    if force and logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    # Консольний хендлер
    console_exists = any(_handler_is_console(h) for h in logger.handlers)
    if not console_exists:
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(desired_level)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    elif update_existing:
        for h in logger.handlers:
            if _handler_is_console(h):
                h.setLevel(desired_level)
                h.setFormatter(formatter)

    # Файловий хендлер (опційний)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = None
        for h in logger.handlers:
            if _handler_is_file(h, log_path):
                file_handler = h
                break

        if file_handler is None:
            fh = RotatingFileHandler(
                filename=str(log_path),
                mode="a",
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
                delay=True,  # файл відкриється при першому записі
            )
            fh.setLevel(desired_level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        elif update_existing:
            file_handler.setLevel(desired_level)
            file_handler.setFormatter(formatter)

    logger.debug(
        "Logger initialized (name=%s, level=%s, file=%s, utc=%s, max_bytes=%s, backups=%s)",
        logger_name, level, str(log_file), use_utc, max_bytes, backup_count
    )
    return logger