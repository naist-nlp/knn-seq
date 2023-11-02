from typing import Any, List

from transformers import AutoTokenizer
from transformers.tokenization_utils import BatchEncoding


class HFTokenizer:
    """Huggingface tokenizer wrapper class.

    Args:
        tokenizer (Any): wrapped tokenizer or dictionary class.
        device (str): the device to place tensors.
    """

    def __init__(self, tokenizer: Any, device: str = "cpu") -> None:
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def build_tokenizer(cls, name_or_path: str, device: str = "cpu") -> "HFTokenizer":
        """Builds a tokenizer.

        Args:
            name_or_path (str): model name or path.
            device (str): the device to place tensors.

        Returns:
            HFTokenizer: this class.
        """
        tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        return cls(tokenizer, device=device)

    def encode(self, line: str) -> List[int]:
        """Encodes a line.

        Args:
            line (str): a line.

        Returns:
            List[int]: a sequence of token index.
        """
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
        batch = []
        for sample in samples:
            batch.append(
                self.tokenizer.prepare_for_model(
                    sample,
                    None,
                    add_spenical_tokens=True,
                    padding=False,
                    truncation=True,
                    pad_to_multiple_of=None,
                    return_attention_mask=False,
                    return_tensors=None,
                )
            )
        return self.tokenizer.pad(batch, padding=True, return_tensors="pt").to(
            self.device
        )
