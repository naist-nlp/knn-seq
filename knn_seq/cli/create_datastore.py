#!/usr/bin/env python3

import logging
import os.path
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from time import time
from typing import Iterator, List

import numpy as np
import torch
from tqdm import tqdm

from knn_seq import utils
from knn_seq.data import Datastore, TokenStorage
from knn_seq.models import build_hf_model

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
    parser.add_argument("--max-tokens", metavar="N", type=int, default=512,
                        help="max tokens per batch")
    parser.add_argument("--save-freq", metavar="N", type=int, default=512,
                        help="save frequency to reduce random write access;\n"
                             "if set a lower value, the less memory will be used but saving will be slower.")
    parser.add_argument("--cpu", action="store_true",
                        help="only use CPU")
    parser.add_argument("--fp16", action="store_true",
                        help="use FP16")
    parser.add_argument("--feature", metavar="TYPE", default="avg",
                        help="specify the feature type (default: avg)\n"
                             "  - avg: averaging the last layer's hidden states\n"
                             "  - cls: [CLS] token\n"
                             "  - sbert: using sentence-transformers")
    parser.add_argument("model_name", metavar="MODEL_NAME",
                        help="model/tokenizer name for huggingface models")
    # fmt: on
    return parser.parse_args()


def main(args):
    logger.info(args)

    use_gpu = torch.cuda.is_available() and not args.cpu
    val = TokenStorage.load(args.outdir)
    model = build_hf_model(args.model_name, args.feature)
    if use_gpu:
        model = model.cuda()
        if args.fp16:
            model = model.half()
    model.tokenizer.pretokenized = True

    datastore_path = os.path.join(args.outdir, "datastore.{}.bin".format(args.feature))
    with Datastore.open(
        datastore_path,
        len(val),
        dim=model.get_embed_dim(),
        dtype=np.float16 if args.fp16 else np.float32,
    ) as mmap:
        logger.info("Creating the datastore to {}".format(datastore_path))
        start_time = time()
        feature_vectors = []
        for i, seq_ids in enumerate(
            tqdm(
                list(batch_sampler(val, max_tokens=args.max_tokens)),
                desc="Datastore creation",
            ),
        ):
            batch = [val[idx].tolist() for idx in seq_ids]
            net_outputs = model(model.tokenizer.collate(batch))
            feature_vectors.append(net_outputs.cpu().numpy())

            if (i + 1) % args.save_freq == 0:
                mmap.add(np.concatenate(feature_vectors))
                feature_vectors = []

        if len(feature_vectors) > 0:
            mmap.add(np.concatenate(feature_vectors))

        end_time = time()

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
