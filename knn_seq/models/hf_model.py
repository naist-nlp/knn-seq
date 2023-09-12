import re
from typing import Dict

import torch.nn as nn
from sentence_transformers.SentenceTransformer import SentenceTransformer
from torch import Tensor
from transformers import AutoModel

from knn_seq.models.hf_tokenizer import HFTokenizer


class HFModelBase(nn.Module):
    """Huggingface model wrapper.

    Args:
        name_or_path (str): model name or path.
    """

    def __init__(self, name_or_path: str) -> None:
        super().__init__()
        self.name_or_path = name_or_path
        self.model = AutoModel.from_pretrained(name_or_path)
        self.init_model()
        self.tokenizer = HFTokenizer.build_tokenizer(name_or_path)

    def init_model(self) -> None:
        for p in self.parameters():
            if getattr(p, "requires_grad", None) is not None:
                p.requires_grad = False
        self.eval()

    def cuda(self, device=None):
        self.tokenizer.device = f"cuda:{device}" or "cpu"
        return super().cuda(device=device)

    def forward(self, *args, **kwargs) -> Tensor:
        """Returns extracted features.

        Returns:
            Tensor: output tensor.
        """
        raise NotImplementedError

    def forward_model(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_embed_dim(self) -> int:
        """Gets the embedding dimension size.

        Returns:
            int: the embedding dimension size.
        """
        return self.model.get_input_embeddings().embedding_dim


class HFModelAvg(HFModelBase):
    def forward(self, net_inputs: Dict[str, Tensor]) -> Tensor:
        """Returns extracted features.

        Args:
            net_inputs (Dict[str, Tensor]): huggingface-format model inputs.

        Returns:
            Tensor: output tensor.
        """
        net_outputs = self.forward_model(**net_inputs)
        non_pad_mask = net_inputs["attention_mask"]
        active_hiddens = net_outputs["last_hidden_state"] * non_pad_mask.unsqueeze(-1)
        return active_hiddens.sum(dim=1) / non_pad_mask.sum(dim=1, keepdim=True)


class HFModelCls(HFModelBase):
    def forward(self, net_inputs: Dict[str, Tensor]) -> Tensor:
        """Returns extracted features.

        Args:
            net_inputs (Dict[str, Tensor]): huggingface-format model inputs.

        Returns:
            Tensor: output tensor.
        """
        net_outputs = self.forward_model(**net_inputs)
        return net_outputs["pooler_output"]


class SentenceTransformerModel(HFModelBase):
    def __init__(self, name_or_path: str) -> None:
        super(HFModelBase, self).__init__()
        self.name_or_path = name_or_path
        self.model = SentenceTransformer(
            re.sub(r"^sentence-transformers/", "", name_or_path)
        )
        self.init_model()
        self.tokenizer = HFTokenizer(self.model.tokenizer)

    def forward(self, net_inputs: Dict[str, Tensor]) -> Tensor:
        """Returns extracted features.

        Args:
            net_inputs (Dict[str, Tensor]): huggingface-format model inputs.

        Returns:
            Tensor: output tensor.
        """
        net_outputs = self.forward_model(net_inputs)
        return net_outputs["sentence_embedding"]

    def get_embed_dim(self) -> int:
        """Gets the embedding dimension size.

        Returns:
            int: the embedding dimension size.
        """
        return self.model.get_sentence_embedding_dimension()


def build_hf_model(model_name: str, feature: str) -> HFModelBase:
    """Build a huggingface model from a model name and a feature type.

    Args:
        model_name (str): model name.
        feature (str): feature type.

    Returns:
        HFModelBase: wrappered huggingface model.
    """
    if model_name.startswith("sentence-transformers/"):
        return SentenceTransformerModel(model_name)
    elif feature == "avg":
        return HFModelAvg(model_name)
    elif feature == "cls":
        return HFModelCls(model_name)
    raise NotImplementedError
