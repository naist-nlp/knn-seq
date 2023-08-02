# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os.path
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from fairseq import utils as fairseq_utils
from fairseq.data import LanguagePairDataset
from fairseq.dataclass import ChoiceEnum
from fairseq.tasks import register_task
from fairseq.tasks.translation import (
    TranslationConfig,
    TranslationTask,
    load_langpair_dataset,
)
from omegaconf import II
from torch import LongTensor

from knn_seq.data import TokenStorage
from knn_seq.dataset_wrapper import (
    LanguagePairDatasetWithOriginalOrder,
    LanguagePairDatasetWithRawSentence,
)
from knn_seq.models import FairseqKNNModel, FairseqSubsetKNNModel, build_hf_model
from knn_seq.models.fairseq_knn_model_base import FairseqKNNModelBase
from knn_seq.search_index import FaissIndex, load_index

logging.basicConfig(
    format="| %(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.getenv("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
    force=True,
)

logger = logging.getLogger(__name__)


SENTENCE_INDEX_CHOICES = ChoiceEnum(["avg", "cls", "senttr", "enc"])
INDEX_METRIC_CHOICES = ChoiceEnum(["l2", "ip", "cos"])


@dataclass
class TranslationKnnConfig(TranslationConfig):
    knn_key: str = field(default="ffn_in", metadata={"help": "Type of kNN key."})
    knn_metric: INDEX_METRIC_CHOICES = field(
        default="l2", metadata={"help": "Distance function for kNN."}
    )
    knn_topk: int = field(
        default=16, metadata={"help": "Retrieve top-k nearest neighbors."}
    )
    knn_nprobe: int = field(
        default=32,
        metadata={
            "help": "Retrieve tokens from nprobe-nearest clusters."
            "This option is only used when using IVF index."
        },
    )
    knn_efsearch: int = field(
        default=64,
        metadata={"help": "This option is only used when using HNSW search."},
    )
    knn_index_path: str = field(default="", metadata={"help": "Path to the kNN index."})
    knn_value_path: str = field(
        default="", metadata={"help": "Path to the values of datastore."}
    )

    src_key: str = field(
        default="senttr", metadata={"help": "Type of source-side key."}
    )
    src_metric: INDEX_METRIC_CHOICES = field(
        default="l2", metadata={"help": "Distance function for source-side kNN."}
    )
    src_knn_model: Optional[str] = field(
        default=None, metadata={"help": "Source kNN model."}
    )
    src_topk: int = field(
        default=5,
        metadata={"help": "Retrieve top-k nearest neighbor sentences."},
    )
    src_nprobe: int = field(
        default=64,
        metadata={
            "help": "Retrieve sentences from nprobe-nearest clusters. "
            "This option is only used when using IVF index."
        },
    )
    src_efsearch: int = field(
        default=64,
        metadata={"help": "This option is only used when using HNSW search."},
    )
    src_index_path: str = field(
        default="", metadata={"help": "Path to the source-side kNN index."}
    )
    src_value_path: str = field(
        default="", metadata={"help": "Path to the source-side values of datastore."}
    )

    knn_threshold: Optional[float] = field(
        default=None,
        metadata={
            "help": "Drop out retrieved neighbors that have lower scores than the threshold."
        },
    )
    knn_weight: float = field(
        default=0.0, metadata={"help": "kNN weight parameter for interpolation"}
    )
    knn_temperature: float = field(
        default=100.0, metadata={"help": "kNN softmax temperature."}
    )
    knn_cpu: bool = field(default=False, metadata={"help": "use CPU to retrieve"})
    knn_ensemble: bool = field(
        default=False, metadata={"help": "Retrieve kNN in each model"}
    )
    knn_gpuivf: bool = field(default=False, metadata={"help": "use IVF on GPU."})
    fp16: bool = II("common.fp16")


@register_task("translation_knn", dataclass=TranslationKnnConfig)
class TranslationKnnTask(TranslationTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: TranslationKnnConfig

    def load_dataset(self, split, epoch=1, combine=False, shuffle=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = fairseq_utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        dataset = LanguagePairDatasetWithOriginalOrder(
            load_langpair_dataset(
                data_path,
                split,
                src,
                self.src_dict,
                tgt,
                self.tgt_dict,
                combine=combine,
                dataset_impl=self.cfg.dataset_impl,
                upsample_primary=self.cfg.upsample_primary,
                left_pad_source=False,
                left_pad_target=False,
                max_source_positions=self.cfg.max_source_positions,
                max_target_positions=self.cfg.max_target_positions,
                load_alignments=self.cfg.load_alignments,
                truncate_source=self.cfg.truncate_source,
                num_buckets=self.cfg.num_batch_buckets,
                shuffle=shuffle and (split != "test"),
                pad_to_multiple=self.cfg.required_seq_len_multiple,
            )
        )
        src_sents_path = os.path.join(
            data_path, "orig", "{}.{}-{}.{}".format(split, src, tgt, src)
        )
        src_sents = None
        if os.path.exists(src_sents_path):
            with open(src_sents_path, mode="r") as f:
                src_sents = [line.strip() for line in f]
        dataset = LanguagePairDatasetWithRawSentence(dataset, src_sents=src_sents)
        self.datasets[split] = dataset

    def build_dataset_for_inference(
        self,
        src_tokens: List[LongTensor],
        src_lengths: List[LongTensor],
        tgt_tokens: Optional[LongTensor] = None,
        constraints=None,
        src_sents: Optional[List[str]] = None,
    ):
        return LanguagePairDatasetWithRawSentence(
            LanguagePairDatasetWithOriginalOrder(
                LanguagePairDataset(
                    src_tokens,
                    src_lengths,
                    self.source_dictionary,
                    tgt=tgt_tokens,
                    tgt_dict=self.target_dictionary,
                    left_pad_source=False,
                    left_pad_target=False,
                    shuffle=False,
                    constraints=constraints,
                )
            ),
            src_sents=src_sents,
        )

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
        prefix_allowed_tokens_fn=None,
    ):
        """
        Build a :class:`~fairseq.SequenceGenerator` instance for this
        task.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            args (fairseq.dataclass.configs.GenerationConfig):
                configuration object (dataclass) for generation
            extra_gen_cls_kwargs (Dict[str, Any]): extra options to pass
                through to SequenceGenerator
            prefix_allowed_tokens_fn (Callable[[int, torch.Tensor], List[int]]):
                If provided, this function constrains the beam search to
                allowed tokens only at each step. The provided function
                should take 2 arguments: the batch ID (`batch_id: int`)
                and a unidimensional tensor of token ids (`inputs_ids:
                torch.Tensor`). It has to return a `List[int]` with the
                allowed tokens for the next generation step conditioned
                on the previously generated tokens (`inputs_ids`) and
                the batch ID (`batch_id`). This argument is useful for
                constrained generation conditioned on the prefix, as
                described in "Autoregressive Entity Retrieval"
                (https://arxiv.org/abs/2010.00904) and
                https://github.com/facebookresearch/GENRE.
        """

        if self.cfg.knn_weight <= 0.0:
            return super().build_generator(
                models,
                args,
                seq_gen_cls=seq_gen_cls,
                extra_gen_cls_kwargs=extra_gen_cls_kwargs,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )

        subset_knn = self.cfg.src_knn_model is not None or self.cfg.src_key == "enc"

        knn_cuda = torch.cuda.is_available() and not self.cfg.knn_cpu
        knn_index_paths = fairseq_utils.split_paths(self.cfg.knn_index_path)
        knn_value_dir = os.path.dirname(self.cfg.knn_value_path)
        if not self.cfg.knn_ensemble:
            knn_index_paths = [knn_index_paths[0]]

        val = TokenStorage.load(knn_value_dir)
        indexes = []
        for path in knn_index_paths:
            index = load_index(path)
            index.set_nprobe(self.cfg.knn_nprobe)
            index.set_efsearch(self.cfg.knn_efsearch)
            if self.cfg.knn_gpuivf and knn_cuda:
                index.to_gpu_search()
            indexes.append(index)
        logger.info(f"Loaded kNN index from {','.join(knn_index_paths)}")

        # naive kNN-MT
        if not subset_knn:
            models = FairseqKNNModel(
                models,
                key=self.cfg.knn_key,
                knn_ensemble=self.cfg.knn_ensemble,
            )
            models.set_index(
                val,
                indexes,
                knn_topk=self.cfg.knn_topk,
                knn_temperature=self.cfg.knn_temperature,
                knn_threshold=self.cfg.knn_threshold,
                knn_weight=self.cfg.knn_weight,
            )
            return super().build_generator(
                models,
                args,
                seq_gen_cls=seq_gen_cls,
                extra_gen_cls_kwargs=extra_gen_cls_kwargs,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            )

        # subset kNN-MT
        if self.cfg.src_key == "enc":
            src_knn_model = None
        else:
            src_knn_model = build_hf_model(self.cfg.src_knn_model, self.cfg.src_key)
            if knn_cuda:
                src_knn_model = src_knn_model.cuda()
                if self.cfg.fp16:
                    src_knn_model = src_knn_model.half()
            logger.info(f"Loaded source kNN model: {src_knn_model}")
        src_value_dir = os.path.dirname(self.cfg.src_value_path)
        src_val = TokenStorage.load(src_value_dir)
        src_index = FaissIndex.load(self.cfg.src_index_path)
        src_index.set_nprobe(self.cfg.src_nprobe)
        src_index.set_efsearch(self.cfg.src_efsearch)
        if knn_cuda:
            src_index.to_gpu_search()
        logger.info(f"Loaded source index from {self.cfg.src_index_path}")

        models = FairseqSubsetKNNModel(
            models,
            key=self.cfg.knn_key,
            knn_ensemble=self.cfg.knn_ensemble,
        )
        models.set_index(
            val,
            indexes,
            src_knn_model,
            src_val,
            src_index,
            knn_topk=self.cfg.knn_topk,
            knn_temperature=self.cfg.knn_temperature,
            knn_threshold=self.cfg.knn_threshold,
            knn_weight=self.cfg.knn_weight,
            src_topk=self.cfg.src_topk,
            use_gpu=knn_cuda,
            use_fp16=self.cfg.fp16,
        )
        return super().build_generator(
            models,
            args,
            seq_gen_cls=seq_gen_cls,
            extra_gen_cls_kwargs=extra_gen_cls_kwargs,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        model = generator.model
        if (
            isinstance(model, FairseqKNNModelBase)
            and sample.get("src_sents", None) is not None
        ):
            model.set_src_sents(sample["src_sents"])

        results = super().inference_step(
            generator,
            models,
            sample,
            prefix_tokens=prefix_tokens,
            constraints=constraints,
        )
        if isinstance(model, FairseqKNNModel):
            model.knn_timer.log_time("kNN")
        if isinstance(model, FairseqSubsetKNNModel):
            model.src_knn_timer.log_time("Source search")
            model.reorder_timer.log_time("Reorder")
            for i, sample_id in enumerate(sample["id"].tolist()):
                src_knn_i = model.src_knn[i]
                print("R-{}\t{}".format(sample_id, " ".join(map(str, src_knn_i))))

        return results
