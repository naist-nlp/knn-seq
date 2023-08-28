#!/usr/bin/env python3

import ast
import concurrent.futures
import copy
import logging
import os
import sys
import time
from argparse import Namespace
from collections import deque
from threading import Lock
from typing import Any, Dict

import fairseq.utils as fairseq_utils
import numpy as np
import torch
from fairseq import checkpoint_utils, options, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
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
        model.prepare_for_inference_(cfg)

    model = FairseqKNNModel(models, key=task.cfg.knn_key)
    if use_cuda:
        model_replicas = [copy.deepcopy(model).cuda(i) for i in range(args.num_gpus)]
        if cfg.common.fp16:
            model_replicas = [m.half() for m in model_replicas]
    else:
        model_replicas = [model]

    num_replicas = len(model_replicas)

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
            task.cfg.src_key if args.store_src_sent else task.cfg.knn_key,
        )
        for i in range(len(models))
    ]
    datastore_paths = [os.path.join(task.cfg.data, fname) for fname in datastore_fnames]
    size = len(dataset) if args.store_src_sent else sum(dataset.tgt_sizes)
    dtype = np.float16 if cfg.common.fp16 else np.float32

    dims = model.get_embed_dim()
    datastores = [
        Datastore._open(
            path, size, dim, dtype, readonly=False, compress=args.compress_datastore
        )
        for path, dim in zip(datastore_paths, dims)
    ]

    def _add_examples(
        rank: int,
        model: FairseqKNNModel,
        batch: Dict[str, Any],
        begin: int,
        end: int,
    ) -> int:
        if use_cuda:
            batch = fairseq_utils.move_to_cuda(batch, f"cuda:{rank}")
        net_input = batch["net_input"]
        orig_order = batch["orig_order"]
        src_tokens = net_input["src_tokens"].index_select(0, orig_order)
        src_lengths = net_input["src_lengths"].index_select(0, orig_order)
        prev_output_tokens = net_input["prev_output_tokens"].index_select(0, orig_order)
        net_outputs = model(
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
        for keys, ds in zip(net_outputs, datastores):
            ds.write_range(keys.cpu().numpy(), begin, end)
        return rank

    logger.info(f"Creating the datastore to {','.join(datastore_paths)}")
    logger.info(f"Datastore size: {size:,}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_replicas) as executor:
        # - `workers` are used to keep track of the reference to futures.
        # - `empties` are used to keep track of actual worker instance and
        #   assign a worker to an empty device.
        workers = set()
        empties = deque(range(num_replicas))
        wp = 0
        start_time = time.perf_counter()
        for batch in tqdm(epoch_iter):
            if args.store_src_sent:
                length = len(batch["id"])
            else:
                length = (
                    batch["net_input"]["prev_output_tokens"]
                    .ne(tgt_dict.pad())
                    .sum()
                    .item()
                )
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
        "--num-gpus", type=int, default=1, help="number of GPUs to compute keys"
    )
    parser.add_argument(
        "--store-src-sent",
        action="store_true",
        help="stores the features of the source sentences",
    )
    parser.add_argument(
        "--compress-datastore", action="store_true", help="compress the datastore"
    )
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == "__main__":
    cli_main()
