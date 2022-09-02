#!/usr/bin/env python3

import ast
import logging
import os
import sys
import time
from argparse import Namespace
from collections import defaultdict
from typing import DefaultDict, List

import fairseq.utils as fairseq_utils
import numpy as np
import torch
from fairseq import checkpoint_utils, options, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from torch import Tensor
from tqdm import tqdm

from knn_seq.data.datastore import Datastore
from knn_seq.models import FairseqKNNModel
from knn_seq.translation_knn import TranslationKnnTask

logging.basicConfig(
    format="| %(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger(__name__)


def main(args: Namespace):
    cfg = convert_namespace_to_omegaconf(args)

    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.batch_size = 1

    logger.info(cfg)

    # Setup task, e.g., translation
    task: TranslationKnnTask = tasks.setup_task(cfg.task)
    tgt_dict = task.target_dictionary

    # Load ensemble
    overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    logger.info("Loading model(s) from {}".format(cfg.common_eval.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        fairseq_utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # Initialize model
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu
    for model in models:
        if model is None:
            continue
        if use_cuda:
            model.cuda()
        model.prepare_for_inference_(cfg)

    model = FairseqKNNModel(models, key=task.cfg.tgt_index_key)
    if use_cuda:
        model = model.cuda()
        if cfg.common.fp16:
            model = model.half()

    # Load the dataset
    task.load_dataset("train")
    dataset = task.dataset("train")
    epoch_iter = task.get_batch_iterator(
        dataset,
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
    ).next_epoch_itr(shuffle=False)

    datastore_fnames = [
        "datastore{}.{}.bin".format(
            "" if i == 0 else i,
            task.cfg.src_index_key if args.store_src_sent else task.cfg.tgt_index_key,
        )
        for i in range(len(models))
    ]
    datastore_paths = [os.path.join(task.cfg.data, fname) for fname in datastore_fnames]
    size = len(dataset) if args.store_src_sent else sum(dataset.tgt_sizes)
    dtype = np.float16 if cfg.common.fp16 else np.float32

    dims = model.get_embed_dim()
    datastores = [
        Datastore._open(path, size, dim, dtype)
        for path, dim in zip(datastore_paths, dims)
    ]
    logger.info(f"Creating the datastore to {','.join(datastore_paths)}")
    start_time = time.perf_counter()
    feature_vectors: DefaultDict[int, List[Tensor]] = defaultdict(list)
    for i, batch in enumerate(tqdm(epoch_iter)):
        if use_cuda:
            batch = fairseq_utils.move_to_cuda(batch)
        net_input = batch["net_input"]
        orig_order = batch["orig_order"]
        src_tokens = net_input["src_tokens"].index_select(0, orig_order)
        src_lengths = net_input["src_lengths"].index_select(0, orig_order)
        prev_output_tokens = net_input["prev_output_tokens"].index_select(0, orig_order)
        net_outputs = model.forward(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            prev_output_tokens=prev_output_tokens,
            output_encoder_features=args.store_src_sent,
        )
        if not args.store_src_sent:
            net_outputs = [
                decoder_out[prev_output_tokens.ne(tgt_dict.pad())]
                for decoder_out in net_outputs
            ]

        for m, output in enumerate(net_outputs):
            feature_vectors[m].append(output.cpu())

        if (i + 1) % args.save_freq == 0:
            for m, ds in enumerate(datastores):
                ds.add(torch.cat(feature_vectors[m]).numpy())
                feature_vectors[m] = []

    if len(feature_vectors) > 0:
        for m, ds in enumerate(datastores):
            ds.add(torch.cat(feature_vectors[m]).numpy())
            feature_vectors[m] = []

    end_time = time.perf_counter()

    for ds in datastores:
        ds.close()

    logger.info(
        "Processed {:,} datapoints in {:.1f} seconds".format(
            size, end_time - start_time
        )
    )
    logger.info("Done")


def cli_main():
    parser = options.get_interactive_generation_parser("translation_knn")
    parser.add_argument(
        "--save-freq", metavar="N", default=512, type=int, help="save frequency"
    )
    parser.add_argument(
        "--store-src-sent",
        action="store_true",
        help="stores the features of the source sentences",
    )
    args = options.parse_args_and_arch(parser)
    args.task = "translation_knn"
    main(args)


if __name__ == "__main__":
    cli_main()
