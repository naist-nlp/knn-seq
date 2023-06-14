#!/usr/bin/env python3

import logging
import os
import sys
from argparse import ArgumentParser, RawTextHelpFormatter

from knn_seq import utils
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
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("--input", "-i", metavar="FILE", required=True,
                        help="input file")
    parser.add_argument("--outdir", "-o", metavar="DIR", default="index",
                        help="output directory")
    parser.add_argument("--buffer-size", "-b", metavar="BUFSIZE", type=int, default=100000)
    parser.add_argument("--num-workers", "-n", metavar="N", type=int, default=8)
    parser.add_argument("--pretokenized", action="store_true",
                        help="if set, word and sub-word tokenizers are not used and only encode tokens in vocabulary IDs.")
    parser.add_argument("model_name_or_path", metavar="MODEL",
                        help="model name or path")
    # fmt: on
    return parser.parse_args()


def main(args):
    logger.info(args)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

    tokenizer = HFTokenizer.build_tokenizer(
        args.model_name_or_path, pretokenized=args.pretokenized
    )

    logger.info("Binarize the text.")
    encoded_lines = []
    for lines in utils.parallel_apply(
        tokenizer.encode_lines,
        utils.read_lines(args.input, args.buffer_size, progress=True),
        num_workers=args.num_workers,
    ):
        encoded_lines.extend(lines)
    val = TokenStorage.binarize(encoded_lines)
    val.save(args.outdir)


def cli_main():
    main(parse_args())


if __name__ == "__main__":
    cli_main()
