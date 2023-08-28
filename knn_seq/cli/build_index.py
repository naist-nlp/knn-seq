#!/usr/bin/env python3

import logging
import math
import os
import sys
import time
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter

import torch
from tqdm import tqdm

from knn_seq.data import Datastore
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
    parser.add_argument("--datastore-path", "-d", metavar="DIR", required=True,
                        help="Path to a datastore file.")
    parser.add_argument("--index-path-prefix", metavar="PREFIX", required=True,
                        help="Path prefix to an output index file."
                             "Output file name is added '.bin' to this argument.")
    parser.add_argument("--cpu", action="store_true",
                        help="only use CPU")
    parser.add_argument("--backend", metavar="NAME", type=str, default="faiss",
                        help="search engine backend")
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
    parser.add_argument("--save-freq", metavar="N", type=int, default=-1,
                        help="Save an index every N times.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Displays the verbose messages.")
    parser.add_argument("--save-freq", type=int, metavar="N", default=-1,
                        help="Save an index every N times.")
    # fmt: on
    return parser.parse_args()


def train_index(args: Namespace, ds: Datastore, use_gpu: bool = False) -> SearchIndex:
    num_data, dim = ds.shape
    train_size = min(args.train_size, num_data)
    index: SearchIndex = build_index(
        args.backend,
        args.metric,
        dim,
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

    trained_index_path = args.index_path_prefix + ".trained.bin"
    if os.path.exists(trained_index_path):
        trained_index = load_index(trained_index_path)
        if trained_index.config == index.config:
            logger.info(f"Load trained index: {trained_index_path}")
            return trained_index
        raise FileExistsError(trained_index_path)

    if use_gpu:
        index.to_gpu_train()
    logger.info(f"Training from {train_size:,} datapoints")
    start_time = time.perf_counter()
    index.train(ds[:train_size], verbose=args.verbose)
    end_time = time.perf_counter()
    logger.info("Training done in {:.1f} seconds".format(end_time - start_time))
    index.save(trained_index_path)
    return load_index(trained_index_path)


def main(args: Namespace):
    logger.info(args)

    datastore_path = args.datastore_path
    index_path = args.index_path_prefix + ".bin"

    if os.path.exists(index_path):
        logger.info(f"{index_path} already exists, skip.")

    use_gpu = torch.cuda.is_available() and not args.cpu
    chunk_size = args.chunk_size
    with Datastore.open(datastore_path) as ds:
        num_data = ds.shape[0]
        index = train_index(args, ds, use_gpu=use_gpu)

        if use_gpu:
            index.to_gpu_add(fp16=ds.is_fp16)

        logger.info(f"Creating a kNN index in {index_path}")
        start_time = time.perf_counter()
        for i in tqdm(range(math.ceil(num_data / chunk_size)), desc="Building index"):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, num_data)
            nadd = end_idx - start_idx
            logger.info(f"Add vectors: {nadd:,} / {num_data:,}")
            index.add(ds[start_idx:end_idx])
            logger.info(f"Index size: {len(index):,}")

            if args.save_freq > 0 and (i + 1) % args.save_freq == 0:
                save_start_time = time.perf_counter()
                index.save(index_path)
                save_end_time = time.perf_counter()
                logger.info(
                    "Saved index in {:.1f} seconds.".format(save_end_time - save_start_time)
                )

    index.save(index_path)
    end_time = time.perf_counter()
    logger.info(f"Added {num_data:,} datapoints")
    logger.info(f"Index size: {len(index):,}")
    logger.info("Created done in {:.1f} seconds".format(end_time - start_time))


def cli_main():
    main(parse_args())


if __name__ == "__main__":
    cli_main()
