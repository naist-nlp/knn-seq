import logging
import time
from typing import List

import fairseq
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


class StopwatchMeter:
    """Stopwatch meter class to computes the sum/avg duration of some event in seconds.

    The original implementation is :code:`fairseq.meters.StopwatchMeter`.
    """

    def __init__(self):
        self.sum = 0
        self.n = 0
        self.start_time = None
        self.stop_time = None

    def start(self):
        self.start_time = time.perf_counter()
        self.stop_time = None

    def stop(self, n: int = 1, prehook=None):
        if self.stop_time is not None:
            # already stopped and wasn't started again
            return

        stop_time = time.perf_counter()
        if self.start_time is not None:
            if prehook is not None:
                prehook()
            self.stop_time = stop_time
            delta = stop_time - self.start_time
            self.sum = self.sum + delta
            self.n = fairseq.meters.type_as(self.n, n) + n

    def reset(self):
        self.sum = 0  # cumulative time during which stopwatch was active
        self.n = 0  # total n across all start/stop
        self.stop_time = None
        self.start_time = None

    @property
    def avg(self):
        return self.sum / self.n if self.n > 0 else self.sum

    @property
    def elapsed_time(self):
        if self.start_time is None or self.stop_time is not None:
            return 0.0
        return time.perf_counter() - self.start_time

    @property
    def lap_time(self):
        if self.start_time is None:
            return 0.0
        elif self.stop_time is None:
            return self.elapsed_time
        return self.stop_time - self.start_time

    def log_time(self, label: str):
        logger.info(
            f"[Time] {label}: {self.lap_time:.3f} s, avg: {self.avg:.3f} s, sum: {self.sum:.3f} s"
        )
