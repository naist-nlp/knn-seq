#!/usr/bin/env python3

import os
import sys
from argparse import Namespace, RawDescriptionHelpFormatter

import fairseq_cli.preprocess
from fairseq import options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.tasks.translation import TranslationTask

from knn_seq.data.token_storage import TokenStorage


def call_preprocess(args: Namespace) -> None:
    return fairseq_cli.preprocess.main(args)


def main(args: Namespace):
    call_preprocess(args)

    cfg = convert_namespace_to_omegaconf(args)

    task = tasks.setup_task(cfg.task)
    task.load_dataset("train")
    dataset = task.dataset("train")

    val = TokenStorage.load_from_fairseq_dataset(dataset, tgt=not args.binarize_src)
    val.save(utils.split_paths(task.cfg.data)[0])


def cli_main():
    parser = options.get_parser("Converter")
    options.add_preprocess_args(parser)
    group = parser.add_argument_group("Preprocessing for kNN")
    # fmt: off
    group.add_argument("--binarize-src", action="store_true",
                       help="Bianrizes the source senteces.")
    # fmt: on
    TranslationTask.add_args(parser)

    parser.epilog = f"""\
Example:
  % python {os.path.basename(sys.argv[0])} \\
      -s en -t de \\
      --srcdict dict.en.txt --tgtdict dict.de.txt \\
      --trainpref corpus/train \\
      binarized/index/de
"""
    parser.formatter_class = RawDescriptionHelpFormatter

    args = parser.parse_args()
    args.destdir = args.data
    main(args)


if __name__ == "__main__":
    cli_main()
