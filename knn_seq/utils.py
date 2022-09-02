import fileinput
import logging
import time
from collections import UserDict, UserList
from itertools import chain
from multiprocessing.pool import Pool
from typing import Any, Callable, Iterable, Iterator, List

import fairseq
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

logger = logging.getLogger(__name__)


def buffer_lines(lines: Iterable, buffer_size: int = 10000) -> Iterator[List[str]]:
    """Yields buffered lines.

    Args:
        lines (Iterable): input lines.
        buffer_size (int): buffer size.

    Yields:
        List[str]: buffered lines.
    """
    buffer = []
    for line in lines:
        buffer.append(line)
        if len(buffer) >= buffer_size:
            yield buffer
            buffer = []
    if len(buffer) > 0:
        yield buffer


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

    def progress_bar(buf):
        if progress:
            return tqdm(buf)
        return buf

    with fileinput.input([input], openhook=fileinput.hook_encoded("utf-8")) as f:
        for lines in buffer_lines(progress_bar(f), buffer_size=buffer_size):
            yield lines


def parallel_apply(
    func: Callable, iterable: Iterable, num_workers: int = 1, *args, **kwargs
) -> Iterator[List[Any]]:
    """Applys a function to an iterable object in parallel.

    Args:
        func (Callable): a function to be applied to an iterable object.
        iterable (Iterable): an iterable object
        num_workers (int): number of workers.
        *args: positional arguments of the function.
        *kwargs: keyword arguments of the function.

    Yields:
        List[Any]: the object to which the function is applied.
    """

    def merge_workers(workers):
        return list(chain.from_iterable(res.get() for res in workers))

    if num_workers > 1:
        workers = []
        with Pool(processes=num_workers) as pool:
            for buffer in iterable:
                workers.append(
                    pool.apply_async(func, args=(buffer, *args), kwds=kwargs)
                )
                if len(workers) >= num_workers:
                    yield merge_workers(workers)
                    workers = []

            if len(workers) > 0:
                yield merge_workers(workers)

    else:
        for buffer in iterable:
            yield func(buffer, *args, **kwargs)


def to_ndarray(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.cpu().numpy()
    return np.array(x)


def to_device(item: Any, use_gpu: bool = True) -> Any:
    """Transfers tensors in arbitrary data structres to a specific device.

    Args:
        item (Any): arbitrary data structures.
        use_gpu (bool): if True, tensors in `item` are transfered to GPUs, otherwise to CPUs.

    Returns:
        Any: the object that is trasfered to the device.
    """
    item_type = type(item)
    if torch.is_tensor(item):
        if use_gpu:
            return item.cuda()
        return item.cpu()
    elif isinstance(item, (dict, UserDict)):
        return item_type({k: to_device(v) for k, v in item.items()})
    elif isinstance(item, (list, UserList)):
        return item_type([to_device(x) for x in item])
    else:
        return item_type(item)


def softmax(scores: Tensor):
    """Computes probability distributions.

    Args:
        scores (FloatTensor): scores tensor.

    Returns:
        Tensor: probability distributions.
    """
    return F.softmax(scores, dim=-1, dtype=torch.float32)


def log_softmax(scores: Tensor):
    """Computes log probability distributions.

    Args:
        scores (FloatTensor): scores tensor.

    Returns:
        Tensor: log probability distributions.
    """
    return F.log_softmax(scores, dim=-1, dtype=torch.float32)


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

    def stop(self, n=1, prehook=None):
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
        self.start()

    @property
    def avg(self):
        return self.sum / self.n if self.n > 0 else self.sum

    @property
    def elapsed_time(self):
        if self.start_time is None:
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
