"""TensorBoard metrics logging."""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Wraps TensorBoard SummaryWriter with lazy init."""

    def __init__(self, log_dir: str = "runs"):
        self.log_dir = log_dir
        self._writer = None

    @property
    def writer(self):
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(self.log_dir)
            except ImportError:
                logger.warning("tensorboard not installed, metrics will be skipped")
                self._writer = _NoOpWriter()
        return self._writer

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


class _NoOpWriter:
    """Fallback when tensorboard is not installed."""
    def add_scalar(self, *args, **kwargs): pass
    def flush(self): pass
    def close(self): pass
