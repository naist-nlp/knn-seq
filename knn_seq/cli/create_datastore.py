#!/usr/bin/env python3

import concurrent.futures
import copy
import logging
import os.path
import sys
import time
from argparse import ArgumentParser, RawTextHelpFormatter
from collections import deque
from threading import Lock
from typing import Iterator, List

import numpy as np
import torch
from tqdm import tqdm

from knn_seq.data import Datastore, TokenStorage
from knn_seq.models import build_hf_model
from knn_seq.models.hf_model import HFModelBase

logging.basicConfig(
    format="| %(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


def batch_sampler(
    val: TokenStorage, max_tokens: int = 512
) -> Iterator[List[np.ndarray]]:
    batch = []
    num_tokens, max_len = 0, 0
    for i, seq_len in enumerate(val.lengths):
        batch.append(i)
        if max_len == 0:
            max_len = seq_len
        num_tokens += max_len
        if max_tokens is not None and num_tokens + max_len > max_tokens:
            yield batch
            batch = []
            num_tokens, max_len = 0, 0

    if len(batch) > 0:
        yield batch


def parse_args():
    # fmt: off
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("--outdir", "-o", metavar="DIR", default="index",
                        help="output directory")
    parser.add_argument("--max-tokens", metavar="N", type=int, default=12000,
                        help="max tokens per batch")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="number of GPUs to compute keys")
    parser.add_argument("--cpu", action="store_true",
                        help="only use CPU")
    parser.add_argument("--fp16", action="store_true",
                        help="use FP16")
    parser.add_argument("--feature", metavar="TYPE", default="senttr",
                        help="specify the feature type (default: senttr)\n"
                             "  - avg: averaging the last layer's hidden states\n"
                             "  - cls: [CLS] token\n"
                             "  - senttr: using sentence-transformers")
    parser.add_argument("model_name", metavar="MODEL_NAME",
                        help="model/tokenizer name for huggingface models")
    parser.add_argument("--compress-datastore", action="store_true",
                        help="compress the datastore")
    # fmt: on
    return parser.parse_args()


def main(args):
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu
    model = build_hf_model(args.model_name, args.feature)
    if use_cuda:
        model_replicas = [copy.deepcopy(model).cuda(i) for i in range(args.num_gpus)]
        if args.fp16:
            model_replicas = [m.half() for m in model_replicas]
    else:
        model_replicas = [model]
    num_replicas = len(model_replicas)

    for m in model_replicas:
        m.tokenizer.pretokenized = True

    val = TokenStorage.load(args.outdir)

    def get_batch():
        for indices in tqdm(
            list(batch_sampler(val, max_tokens=args.max_tokens)),
            desc="Datastore creation",
        ):
            yield [val[idx].tolist() for idx in indices]

    datastore_path = os.path.join(args.outdir, "datastore.{}.bin".format(args.feature))
    with Datastore.open(
        datastore_path,
        len(val),
        dim=model.get_embed_dim(),
        dtype=np.float16 if args.fp16 else np.float32,
        readonly=False,
        compress=args.compress_datastore,
    ) as ds:

        def _add_examples(
            lock: Lock, model: HFModelBase, batch: List[List[int]], begin: int, end: int
        ):
            with lock:
                net_inputs = model.tokenizer.collate(batch)
                net_outputs = model(net_inputs)
                ds.write_range(net_outputs.cpu().numpy(), begin, end)

        logger.info(f"Creating the datastore to {datastore_path}")
        logger.info(f"Datastore size: {len(val):,}")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_replicas
        ) as executor:
            locks = [Lock() for _ in range(num_replicas)]
            wp = 0
            workers = set()
            empties = deque(range(num_replicas))
            start_time = time.perf_counter()
            for batch in get_batch():
                length = len(batch)
                rank = empties.popleft()
                workers.add(
                    executor.submit(
                        _add_examples,
                        locks[rank],
                        model_replicas[rank],
                        batch,
                        wp,
                        wp + length,
                    )
                )
                wp += length

                if len(empties) <= 0:
                    # Wait until one worker is finished.
                    workers = concurrent.futures.wait(
                        workers, return_when="FIRST_COMPLETED"
                    )[1]
                    while len(empties) <= 0:
                        # Slight delay in unlocking may occur
                        empties = deque(
                            i for i, lock in enumerate(locks) if not lock.locked()
                        )
                        time.sleep(0.1)

    end_time = time.perf_counter()

    logger.info(
        "Processed {:,} datapoints in {:.1f} seconds".format(
            len(val), end_time - start_time
        )
    )
    logger.info("Done")


def cli_main():
    main(parse_args())


if __name__ == "__main__":
    cli_main()
