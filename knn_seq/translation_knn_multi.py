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
from fairseq.data.multilingual.multilingual_data_manager import (
    MultilingualDatasetManager,
)
from fairseq.data.multilingual.sampling_method import SamplingMethod
from fairseq.dataclass import ChoiceEnum
from fairseq.tasks import LegacyFairseqTask, register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
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


class MultilingualDatasetManagerWrapper(MultilingualDatasetManager):
    @classmethod
    def setup_data_manager(cls, args, lang_pairs, langs, dicts, sampling_method):
        return cls(args, lang_pairs, langs, dicts, sampling_method)

    def load_a_dataset(
        self,
        split,
        data_path,
        src,
        src_dict,
        tgt,
        tgt_dict,
        combine,
        prepend_bos=False,
        langpairs_sharing_datasets=None,
        data_category=None,
        **extra_kwargs,
    ):
        ds = super().load_a_dataset(
            split,
            data_path,
            src,
            src_dict,
            tgt,
            tgt_dict,
            combine,
            prepend_bos=False,
            langpairs_sharing_datasets=None,
            data_category=None,
            **extra_kwargs,
        )
        ds = LanguagePairDatasetWithOriginalOrder(ds)
        src_sents_path = os.path.join(
            data_path, "orig", "{}.{}-{}.{}".format(split, src, tgt, src)
        )
        src_sents = None
        if os.path.exists(src_sents_path):
            with open(src_sents_path, mode="r") as f:
                src_sents = [line.strip() for line in f]
        ds = LanguagePairDatasetWithRawSentence(ds, src_sents=src_sents)
        return ds


@register_task("translation_knn_multi")
class TranslationKnnMultiTask(TranslationMultiSimpleEpochTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        langs (List[str]): a list of languages that are being supported
        dicts (Dict[str, fairseq.data.Dictionary]): mapping from supported languages to their dictionaries
        training (bool): whether the task should be configured for training or not

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    def __init__(self, args, langs, dicts, training):
        LegacyFairseqTask.__init__(self, args)
        self.cfg = args
        self.langs = langs
        self.dicts = dicts
        self.training = training
        if training:
            self.lang_pairs = args.lang_pairs
        else:
            self.lang_pairs = ["{}-{}".format(args.source_lang, args.target_lang)]
        # eval_lang_pairs for multilingual translation is usually all of the
        # lang_pairs. However for other multitask settings or when we want to
        # optimize for certain languages we want to use a different subset. Thus
        # the eval_lang_pairs class variable is provided for classes that extend
        # this class.
        self.eval_lang_pairs = self.lang_pairs
        # model_lang_pairs will be used to build encoder-decoder model pairs in
        # models.build_model(). This allows multitask type of sub-class can
        # build models other than the input lang_pairs
        self.model_lang_pairs = self.lang_pairs
        self.source_langs = [d.split("-")[0] for d in self.lang_pairs]
        self.target_langs = [d.split("-")[1] for d in self.lang_pairs]
        self.check_dicts(self.dicts, self.source_langs, self.target_langs)

        self.sampling_method = SamplingMethod.build_sampler(args, self)
        self.data_manager = MultilingualDatasetManagerWrapper.setup_data_manager(
            args, self.lang_pairs, langs, dicts, self.sampling_method
        )

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        TranslationMultiSimpleEpochTask.add_args(parser)
        # fmt: off
        parser.add_argument("--knn-key", type=str, default="ffn_in", help="Type of kNN key.")
        parser.add_argument("--knn-metric", type=str, default="l2", help="Distance function for kNN.")
        parser.add_argument("--knn-topk", type=int, default=16, help="Retrieve top-k nearest neighbors.")
        parser.add_argument("--knn-nprobe", type=int, default=32,
                            help="Retrieve tokens from nprobe-nearest clusters."
                            "This option is only used when using IVF index.")
        parser.add_argument("--knn-efsearch", type=int, default=64, help="This option is only used when using HNSW search.")
        parser.add_argument("--knn-index-path", type=str, default="", help="Path to the kNN index.")
        parser.add_argument("--knn-value-path", type=str, default="", help="Path to the values of datastore.")
        parser.add_argument("--src-key", type=str, default="senttr", help="Type of source-side key.")
        parser.add_argument("--src-metric", type=str, default="l2", help="Distance function for source-side kNN.")
        parser.add_argument("--src-knn-model", type=str, default=None, help="Source kNN model.")
        parser.add_argument("--src-topk", type=int, default=5,
                            help="Retrieve top-k nearest neighbor sentences.")
        parser.add_argument("--src-nprobe", type=int, default=64,
                            help="Retrieve sentences from nprobe-nearest clusters."
                            "This option is only used when using IVF index.")
        parser.add_argument("--src-efsearch", type=int, default=64,
                            help="This option is only used when using HNSW search.")
        parser.add_argument("--src-index-path", type=str, default="", help="Path to the source-side kNN index.")
        parser.add_argument("--src-value-path", type=str, default="", help="Path to the source-side values of datastore.")
        parser.add_argument("--knn_threshold", type=float, default=None,
                            help="Drop out retrieved neighbors that have lower scores than the threshold.")
        parser.add_argument("--knn-weight", type=float, default=0.0,
                            help="kNN weight parameter for interpolation")
        parser.add_argument("--knn-temperature", type=float, default=100.0,
                            help="kNN softmax temperature.")
        parser.add_argument("--knn-cpu", action="store_true",
                            help="use CPU to retrieve")
        parser.add_argument("--knn-ensemble", action="store_true",
                            help="Retrieve kNN in each model")
        parser.add_argument("--knn-gpuivf", action="store_true", help="use IVF on GPU.")
        # fmt: on

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
        """

        if not getattr(args, "keep_inference_langtok", False):
            _, tgt_langtok_spec = self.args.langtoks["main"]
            if tgt_langtok_spec:
                tgt_lang_tok = self.data_manager.get_decoder_langtok(
                    self.args.target_lang, tgt_langtok_spec
                )
                extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
                extra_gen_cls_kwargs["symbols_to_strip_from_output"] = {tgt_lang_tok}

        if self.args.knn_weight <= 0.0:
            return super().build_generator(
                models,
                args,
                seq_gen_cls=seq_gen_cls,
                extra_gen_cls_kwargs=extra_gen_cls_kwargs,
            )

        subset_knn = self.args.src_knn_model is not None or self.args.src_key == "enc"

        knn_cuda = torch.cuda.is_available() and not self.args.knn_cpu
        knn_index_paths = fairseq_utils.split_paths(self.args.knn_index_path)
        knn_value_dir = os.path.dirname(self.args.knn_value_path)
        if not self.args.knn_ensemble:
            knn_index_paths = [knn_index_paths[0]]

        val = TokenStorage.load(knn_value_dir)
        indexes = []
        for path in knn_index_paths:
            index = load_index(path)
            index.set_nprobe(self.args.knn_nprobe)
            index.set_efsearch(self.args.knn_efsearch)
            if self.args.knn_gpuivf and knn_cuda:
                index.to_gpu_search()
            indexes.append(index)
        logger.info(f"Loaded kNN index from {','.join(knn_index_paths)}")

        # naive kNN-MT
        if not subset_knn:
            models = FairseqKNNModel(
                models,
                key=self.args.knn_key,
                knn_ensemble=self.args.knn_ensemble,
            )
            models.set_index(
                val,
                indexes,
                knn_topk=self.args.knn_topk,
                knn_temperature=self.args.knn_temperature,
                knn_threshold=self.args.knn_threshold,
                knn_weight=self.args.knn_weight,
            )
            return super().build_generator(
                models,
                args,
                seq_gen_cls=seq_gen_cls,
                extra_gen_cls_kwargs=extra_gen_cls_kwargs,
            )

        # subset kNN-MT
        if self.args.src_key == "enc":
            src_knn_model = None
        else:
            src_knn_model = build_hf_model(self.args.src_knn_model, self.args.src_key)
            if knn_cuda:
                src_knn_model = src_knn_model.cuda()
                if args.fp16:
                    src_knn_model = src_knn_model.half()
            logger.info(f"Loaded source kNN model: {src_knn_model}")
        src_value_dir = os.path.dirname(self.args.src_value_path)
        src_val = TokenStorage.load(src_value_dir)
        src_index = FaissIndex.load(self.args.src_index_path)
        src_index.set_nprobe(self.args.src_nprobe)
        src_index.set_efsearch(self.args.src_efsearch)
        if knn_cuda:
            src_index.to_gpu_search()
        logger.info(f"Loaded source index from {self.args.src_index_path}")

        models = FairseqSubsetKNNModel(
            models,
            key=self.args.knn_key,
            knn_ensemble=self.args.knn_ensemble,
        )
        models.set_index(
            val,
            indexes,
            src_knn_model,
            src_val,
            src_index,
            knn_topk=self.args.knn_topk,
            knn_temperature=self.args.knn_temperature,
            knn_threshold=self.args.knn_threshold,
            knn_weight=self.args.knn_weight,
            src_topk=self.args.src_topk,
            use_gpu=knn_cuda,
            use_fp16=args.fp16,
        )
        return super().build_generator(
            models,
            args,
            seq_gen_cls=seq_gen_cls,
            extra_gen_cls_kwargs=extra_gen_cls_kwargs,
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
