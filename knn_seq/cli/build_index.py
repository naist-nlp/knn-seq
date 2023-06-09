#!/usr/bin/env python3

import datetime
import logging
import os
import shutil
import sys
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from time import time

import torch
from tqdm import tqdm

from knn_seq.data import Datastore
from knn_seq.data.token_storage import TokenStorage
from knn_seq.search_index import build_index, load_index
from knn_seq.search_index.search_index import SearchIndex

logging.basicConfig(
    format="| %(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("cli.build_index")


def parse_args():
    parser = ArgumentParser(
        description="Build the index from datastore for efficient search.",
        formatter_class=RawTextHelpFormatter,
    )
    # fmt: off
    parser.add_argument("--dir", "-d", metavar="DIR", required=True,
                        help="input / output directory")
    parser.add_argument("--output", "-o", metavar="NAME",
                        help="output filename")
    parser.add_argument("--outpref", metavar="NAME", type=str, default="index",
                        help="output filename prefix")
    parser.add_argument("--append", metavar="DIR", type=str, default="",
                        help="append vectors to an existing index.")
    parser.add_argument("--cpu", action="store_true",
                        help="only use CPU")
    parser.add_argument("--backend", metavar="NAME", type=str, default="faiss",
                        help="search engine backend")
    parser.add_argument("--feature", metavar="TYPE", default="avg",
                        help="specify the feature type (default: avg)\n"
                             "  - avg: averaging the last layer's hidden state\n"
                             "  - cls: [CLS] token\n"
                             "  - sbert: sentence BERT\n"
                             "  - ffn_out: the decoder's hidden states\n"
                             "  - ffn_in: the decoder's hidden states")
    parser.add_argument("--metric", metavar="TYPE", default="l2",
                        help="specify the distance function (default: l2)\n"
                             "  - l2: L2-norm\n"
                             "  - ip: inner product\n"
                             "  - cos: cosine similarity")
    parser.add_argument("--chunk-size", metavar="N", type=int, default=1000000,
                        help="the number of data to be loaded at a time")
    parser.add_argument("--train-size", metavar="N", type=int, default=1000000,
                        help="the number of training data")
    parser.add_argument("--hnsw-edges", metavar="N", type=int, default=0,
                        help="the number of IVF clusters")
    parser.add_argument("--ivf-lists", metavar="N", type=int, default=0,
                        help="the number of IVF clusters")
    parser.add_argument("--pq-subvec", metavar="N", type=int, default=0,
                        help="the number of PQ subvectors")
    parser.add_argument("--use-opq", action="store_true",
                        help="Use OPQ")
    parser.add_argument("--use-pca", action="store_true",
                        help="Use PCA")
    parser.add_argument("--transform-dim", metavar="N", type=int, default=-1,
                        help="the dimension size of vector transform")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Displays the verbose messages.")
    parser.add_argument("--index-ids", metavar="N", type=int, default=[0], nargs="+",
                        help="Number of indexes to build.")
    parser.add_argument("--safe", action="store_true",
                        help="Saves the index every time data is added.")
    # fmt: on
    return parser.parse_args()


def train_index(args: Namespace, ds: Datastore, index_path: str, use_gpu: bool = False):
    num_data, dim = ds.shape
    train_size = min(args.train_size, num_data)
    index: SearchIndex = build_index(
        args.backend,
        args.metric,
        dim,
        size=num_data,
        hnsw_edges=args.hnsw_edges,
        ivf_lists=args.ivf_lists,
        pq_subvec=args.pq_subvec,
        transform_dim=args.transform_dim,
        use_opq=args.use_opq,
        use_pca=args.use_pca,
    )
    logger.info(index.config)
    logger.info(f"Raw vector: {index.dim} dim")

    if index.is_trained:
        return index

    trained_index_path = os.path.splitext(index_path)[0] + ".trained.bin"
    if os.path.exists(trained_index_path):
        trained_index = load_index(trained_index_path)
        if trained_index.config == index.config:
            logger.info(f"Load trained index: {trained_index_path}")
            return trained_index
        raise FileExistsError(trained_index_path)

    if use_gpu:
        index.to_gpu_train()
    logger.info(f"Training from {train_size:,} datapoints")
    start_time = time()
    index.train(ds[:train_size], verbose=args.verbose)
    end_time = time()
    logger.info("Training done in {:.1f} seconds".format(end_time - start_time))
    index.save(trained_index_path)
    return load_index(trained_index_path)


NOW = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def backup_path(path: str) -> str:
    return path + "." + NOW


def merge_index_values(save_path: str, other_path: str):
    value_path = os.path.join(save_path, "values.bin")
    old_value_path = backup_path(value_path)
    logger.info(f"Copy: {value_path} -> {old_value_path}")
    shutil.copy(value_path, old_value_path)

    token_storage = TokenStorage.load(save_path)
    token_storage.merge(save_path, other_path)


def main(args: Namespace, datastore_path: str, index_path: str):
    use_gpu = torch.cuda.is_available() and not args.cpu
    chunk_size = args.chunk_size
    with Datastore.open(datastore_path, readonly=True) as ds:
        num_data = ds.shape[0]
        if args.append != "":
            if args.safe:
                old_index_path = backup_path(index_path)
                logger.info(f"Copy: {index_path} -> {old_index_path}")
                shutil.copy(index_path, old_index_path)
                if not os.path.exists(old_index_path):
                    raise FileNotFoundError(f"Copy failed: {old_index_path}")
            index = load_index(index_path)
        else:
            index = train_index(args, ds, index_path, use_gpu=use_gpu)

        if use_gpu:
            index.to_gpu_add(fp16=ds.is_fp16)

        logger.info(f"Creating the feature index in {index_path}")
        start_time = time()
        for i in tqdm(range(num_data // chunk_size), desc="Building index"):
            offset = i * chunk_size
            nadd = offset + chunk_size
            logger.info(f"Add vectors: {nadd:,} / {num_data:,}")
            index.add(ds[offset : offset + chunk_size], verbose=args.verbose)
            logger.info(f"Index size: {len(index):,}")
            if args.append == "" and args.safe:
                logger.info("Save index")
                index.save(index_path)
        if num_data % chunk_size > 0:
            offset = (num_data // chunk_size) * chunk_size
            index.add(ds[offset:])

        logger.info(f"Saving the index")
        index.save(index_path)

        end_time = time()
        logger.info(f"Added {num_data:,} datapoints")
        logger.info(f"Index size: {len(index):,}")
        logger.info("Created done in {:.1f} seconds".format(end_time - start_time))

    if args.append != "":
        logger.info(f"Merge index values: {args.append} -> {args.dir}")
        merge_index_values(args.dir, args.append)
        logger.info(f"Done.")


def call_main(args):
    logger.info(args)

    def save_path(filename):
        return os.path.join(args.dir, filename)

    feature = args.feature
    metric = args.metric

    index_ids = list(args.index_ids)
    for i in index_ids:
        datastore_fname = "datastore{}.{}.bin".format("" if i == 0 else i, feature)
        if args.append == "":
            datastore_path = save_path(datastore_fname)
        else:
            datastore_path = os.path.join(args.append, datastore_fname)

        if getattr(args, "output", None) is not None:
            index_fname = args.output
        else:
            prefix = args.outpref
            if i > 0:
                prefix = args.outpref + str(i)
            index_fname = "{}.{}.{}.bin".format(prefix, feature, metric)
        index_path = save_path(index_fname)

        if os.path.exists(index_path) and args.append == "":
            logger.info(f"{index_path} already exists, skip.")
            continue
        elif not os.path.exists(index_path) and args.append != "":
            logger.info(f"{index_path} not exists, skip.")
            continue

        main(args, datastore_path, index_path)


def cli_main():
    call_main(parse_args())


if __name__ == "__main__":
    cli_main()
