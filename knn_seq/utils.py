import contextlib
import logging
import time
from typing import List

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def pad(tensors: List[Tensor], padding_idx: int) -> Tensor:
    """Pads to tensors.

    Args:
        tensors (List[Tensor]): A tensor list.
        padding_idx (int): Padding index.
    """

    max_len = max(len(t) for t in tensors)
    dtype = tensors[0].dtype
    new_tensor = torch.full(
        (len(tensors), max_len), fill_value=padding_idx, dtype=dtype
    )
    for i, t in enumerate(tensors):
        new_tensor[i, : len(t)] = t
    return new_tensor


class Stopwatch:
    """Stopwatch meter class to computes the duration of some event in seconds.

    Attributes:
        acc (float): Accumulated time.
        lap (float): Lap time of the latest event.

    Example:
        >>> timer = Stopwatch()
        >>> for i in range(10):
                with timer():
                    time.sleep(1)
        >>> print(f"{timer.acc:.3f}")
        10.000
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0
        self.acc = 0.0

    @contextlib.contextmanager
    def __call__(self):
        """Measure the time."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.acc += time.perf_counter() - start
            self.n += 1

    @property
    def avg(self) -> float:
        """Returns the averaged time per event."""
        return self.acc / float(self.n) if self.n > 0 else 0.0

    def log(self, label: str):
        logger.info(
            f"[Time] {label}: acc: {self.acc:.3f} s, avg: {self.avg:.3f} s (n={self.n})"
        )
