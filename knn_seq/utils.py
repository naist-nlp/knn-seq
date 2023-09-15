import fileinput
import logging
import time
from collections import UserDict, UserList
from typing import Any, Iterator, List

import fairseq
import torch
from torch import Tensor
from tqdm import tqdm

logger = logging.getLogger(__name__)


def read_lines(
    input: str, buffer_size: int = 10000, progress: bool = False
) -> Iterator[List[str]]:
    """Reads lines from file or stdin.

    Args:
        input (str): input file path or `-` for stdin.
        buffer_size (int): buffer size.

    Yields:
        List[str]: buffered input lines.
    """
    if buffer_size <= 0:
        raise ValueError("buffer_size must be at least 1")

    def progress_bar(buf):
        if progress:
            return tqdm(buf)
        return buf

    with fileinput.input([input], openhook=fileinput.hook_encoded("utf-8")) as f:
        buffer = []
        for line in progress_bar(f):
            buffer.append(line)
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []
        if len(buffer) > 0:
            yield buffer


def to_device(item: Any, device: str = "cpu") -> Any:
    """Transfers tensors in arbitrary data structres to a specific device.

    Args:
        item (Any): arbitrary data structures.
        device (str): tensors in `item` are transfered to specified devices.

    Returns:
        Any: the object that is trasfered to the device.
    """
    item_type = type(item)
    if torch.is_tensor(item):
        return item.to(device=device)
    elif isinstance(item, (dict, UserDict)):
        return item_type({k: to_device(v, device=device) for k, v in item.items()})
    elif isinstance(item, (list, UserList)):
        return item_type([to_device(x, device=device) for x in item])
    else:
        return item


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
