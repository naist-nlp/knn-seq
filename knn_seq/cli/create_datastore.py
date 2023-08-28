#!/usr/bin/env python3

import concurrent.futures
import copy
import logging
import os.path
import sys
import time
from argparse import ArgumentParser, RawTextHelpFormatter
from collections import deque
from typing import Iterator, List, Optional

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
    val: TokenStorage,
    max_tokens: Optional[int] = None,
    max_sentences: Optional[int] = None,
) -> Iterator[List[np.ndarray]]:
    assert max_tokens is not None or max_sentences is not None
    batch = []
    num_tokens, max_len = 0, 0
    for i, seq_len in enumerate(val.lengths):
        batch.append(i)
        if max_len == 0:
            max_len = seq_len
        num_tokens += max_len
        if (max_sentences is not None and len(batch) >= max_sentences) or (
            max_tokens is not None and num_tokens + max_len > max_tokens
        ):
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
    bsz_group = parser.add_mutually_exclusive_group()
    bsz_group.add_argument("--max-tokens", metavar="N", type=int, default=None,
                           help="max tokens per batch")
    bsz_group.add_argument("--max-sentences", "--batch-size", metavar="N", type=int, default=None,
                           help="max sentences per batch")
    parser.add_argument("--num-gpus", metavar="N", type=int, default=1,
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

    max_tokens: Optional[int] = args.max_tokens
    max_sentences: Optional[int] = args.max_sentences
    if max_tokens is None and max_sentences is None:
        max_sentences = 128

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

    def batch_iterator():
        for indices in tqdm(
            list(batch_sampler(val, max_tokens, max_sentences)),
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
            rank: int, model: HFModelBase, batch: List[List[int]], begin: int, end: int
        ) -> int:
            net_inputs = model.tokenizer.collate(batch)
            net_outputs = model(net_inputs)
            ds.write_range(net_outputs.cpu().numpy(), begin, end)
            return rank

        logger.info(f"Creating the datastore to {datastore_path}")
        logger.info(f"Datastore size: {len(val):,}")
        start_time = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_replicas
        ) as executor:
            wp = 0
            workers = set()
            empties = deque(range(num_replicas))
            for batch in batch_iterator():
                length = len(batch)
                rank = empties.popleft()
                workers.add(
                    executor.submit(
                        _add_examples,
                        rank,
                        model_replicas[rank],
                        batch,
                        wp,
                        wp + length,
                    )
                )
                wp += length

                if len(workers) >= num_replicas:
                    # Wait until one worker is finished.
                    finished, workers = concurrent.futures.wait(
                        workers, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    empties = deque(res.result() for res in finished)

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
