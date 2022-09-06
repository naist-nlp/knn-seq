from typing import List, Optional

import torch
from fairseq.data.base_wrapper_dataset import BaseWrapperDataset
from fairseq.data.dictionary import Dictionary
from fairseq.data.language_pair_dataset import LanguagePairDataset


class LanguagePairDatasetWithRawSentence(BaseWrapperDataset):
    """A wrapper for :class:`fairseq.data.LanguagePairDatset` to get raw sentneces.

    Args:
        dataset: (LanguagePairDataset): wrapped language pair dataset.
    """

    def __init__(
        self,
        dataset: LanguagePairDataset,
        src_sents: Optional[List[str]] = None,
        tgt_sents: Optional[List[str]] = None,
    ):
        super().__init__(dataset)
        self.src_dict: Dictionary = dataset.src_dict
        self.tgt_dict: Dictionary = dataset.tgt_dict
        self.src = dataset.src
        self.src_sizes = dataset.src_sizes
        self.tgt = dataset.tgt
        self.tgt_sizes = dataset.tgt_sizes
        self.src_sents = src_sents
        self.tgt_sents = tgt_sents

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `src_sents` (List[str]): original source sentences
                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """

        res = self.dataset.collater(samples, pad_to_length=pad_to_length)
        if self.src_sents is not None:
            res["src_sents"] = [self.src_sents[i] for i in res["id"]]
        if self.tgt_sents is not None:
            res["tgt_sents"] = [self.tgt_sents[i] for i in res["id"]]
        return res


class LanguagePairDatasetWithOriginalOrder(BaseWrapperDataset):
    """A wrapper for :class:`fairseq.data.LanguagePairDatset` to get the original sort order.

    Args:
        dataset: (LanguagePairDataset): wrapped language pair dataset.
    """

    def __init__(self, dataset: LanguagePairDataset):
        super().__init__(dataset)
        self.src_dict: Dictionary = dataset.src_dict
        self.tgt_dict: Dictionary = dataset.tgt_dict
        self.src = dataset.src
        self.src_sizes = dataset.src_sizes
        self.tgt = dataset.tgt
        self.tgt_sizes = dataset.tgt_sizes

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `orig_order` (LongTensor): original order.
                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """

        res = self.dataset.collater(samples, pad_to_length=pad_to_length)
        src_lengths = torch.LongTensor(
            [s["source"].ne(self.dataset.src_dict.pad()).long().sum() for s in samples]
        )
        sort_order = src_lengths.argsort(descending=True)
        orig_order = sort_order.argsort()
        res["orig_order"] = orig_order
        return res
