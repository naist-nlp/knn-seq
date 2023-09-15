import re
from typing import Any, List, Union

from transformers import AutoTokenizer
from transformers.tokenization_utils import BatchEncoding

from knn_seq import utils

SPACE_NORMALIZER = re.compile(r"\s+")


def space_tokenize(line: str) -> List[str]:
    """Tokenizes a string line by space.

    Args:
        line (str): input line.

    Returns:
        List[str]: space-tokenized line.
    """
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


class HFTokenizer:
    """Huggingface tokenizer wrapper class.

    Args:
        tokenizer (Any): wrapped tokenizer or dictionary class.
        pretokenized (bool): processes as tokenized text.
        device (str): the device to place tensors.
    """

    def __init__(
        self,
        tokenizer: Any,
        pretokenized: bool = False,
        device: str = "cpu",
    ) -> None:
        self.tokenizer = tokenizer
        self.pretokenized = pretokenized
        self.device = device

    @classmethod
    def build_tokenizer(
        cls,
        name_or_path: str,
        pretokenized: bool = False,
        device: str = "cpu",
    ) -> "HFTokenizer":
        """Builds a tokenizer.

        Args:
            name_or_path (str): model name or path.
            pretokenized (bool): processes as tokenized text.
            device (str): the device to place tensors.

        Returns:
            HFTokenizer: this class.
        """
        tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        return cls(tokenizer, pretokenized=pretokenized, device=device)

    def encode(self, line: Union[str, List[str]]) -> List[int]:
        """Encodes a line.

        Args:
            line (Union[str, List[str]]): a line.

        Returns:
            List[int]: a sequence of token index.
        """
        if self.pretokenized:
            if isinstance(line, str):
                return self.tokenizer.convert_tokens_to_ids(space_tokenize(line))
        return self.tokenizer.encode(line, add_special_tokens=False)

    def decode(self, ids: List[int]) -> List[str]:
        """Decodes a token index sequence.

        Args:
            ids (List[int]): a token index sequence.

        Returns:
            List[str]: a sequence of token.
        """
        return self.tokenizer.convert_ids_to_tokens(ids)

    def collate(self, samples: List[List[int]]) -> BatchEncoding:
        """Makes a mini-batch from samples.

        Args:
            samples (List[List[int]]): encoded token sequences.

        Returns:
            BatchEncoding: huggingface model input.
        """
        batch = {}
        for sample in samples:
            item = self.tokenizer.prepare_for_model(
                sample,
                None,
                add_spenical_tokens=True,
                padding=False,
                pad_to_multiple_of=None,
                return_attention_mask=False,
                return_tensors=None,
            )
            for key, value in item.items():
                if key not in batch:
                    batch[key] = []
                batch[key].append(value)

        batch = self.tokenizer.pad(batch, padding=True, return_tensors="pt")
        if self.device != "cpu":
            batch = utils.to_device(batch, device=self.device)
        return BatchEncoding(batch)

    def encode_lines(self, lines: List[str]) -> List[List[int]]:
        """Encodes lines.

        Args:
            lines (List[str]): lines.

        List[List[int]]: sequences of token indices.
        """
        return [self.encode(line) for line in lines]
