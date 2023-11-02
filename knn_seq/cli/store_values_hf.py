#!/usr/bin/env python3
import concurrent.futures
import logging
import os
import sys
from argparse import ArgumentParser

from tqdm import tqdm

from knn_seq.data import TokenStorage
from knn_seq.models.hf_tokenizer import HFTokenizer

logging.basicConfig(
    format="| %(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def parse_args():
    # fmt: off
    parser = ArgumentParser()
    parser.add_argument("--input", "-i", metavar="FILE", required=True,
                        help="input file")
    parser.add_argument("--outdir", "-o", metavar="DIR", default="index",
                        help="output directory")
    parser.add_argument("--buffer-size", "-b", metavar="BUFSIZE", type=int, default=100000,
                        help="buffer size to read lines")
    parser.add_argument("--num-workers", "-n", metavar="N", type=int, default=8,
                        help="number of workers to encode lines")
    parser.add_argument("model_name_or_path", metavar="MODEL",
                        help="model name or path")
    # fmt: on
    return parser.parse_args()


def main(args):
    logger.info(args)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

    tokenizer = HFTokenizer.build_tokenizer(args.model_name_or_path)

    logger.info("Binarize the text.")
    encoded_lines = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.num_workers
    ) as executor:
        with open(args.input, mode="r") as f:
            encoded_lines = list(
                tqdm(executor.map(tokenizer.encode, f, chunksize=args.buffer_size))
            )

    val = TokenStorage.binarize(encoded_lines)
    val.save(args.outdir)


def cli_main():
    main(parse_args())


if __name__ == "__main__":
    cli_main()
