# Standard Library
import math

# Third Party Library
import torch
from torch import nn


class NumberTokenizer:
    def __init__(self, max_num: int) -> None:
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.sep_token_id = 2
        self.mask_token_id = 3

        self.special_token_num = 4
        self.vocab_size = max_num + 1 + self.special_token_num

    def tokenize(self, num_seq: list[int]):
        return [num + 4 for num in num_seq]

    def encode(self, num_seq: list[int]):
        input_ids = (
            [self.cls_token_id]
            + [num + 4 for num in num_seq]
            + [self.sep_token_id]
        )
        return input_ids

    def decode(self, input_ids: list[int]):
        # TODO: remove [CLS] and [SEP]?
        return [num - 4 for num in input_ids[1:-1]]


class PRNGConfig:
    def __init__(
        self,
        vocab_size: int = 100,
        embed_dim: int = 512,
        max_seq_len: int = 10,
        pos_dropout_rate: float = 0.1,
        transformer_dropout_rate: float = 0.1,
        layer_norm_eps: float = 1e-12,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        padding_idx: int = 0,
    ) -> None:
        super(PRNGConfig, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.pos_dropout_rate = pos_dropout_rate
        self.transformer_dropout_rate = transformer_dropout_rate
        self.layer_norm_eps = layer_norm_eps
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.padding_idx = padding_idx

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} {self.to_json_string(use_diff=False)}"
        )


class PosEncoder(nn.Module):
    def __init__(
        self,
        max_seq_len: int,
        embed_dim: int,
        pos_dropout_rate: float,
    ):
        super(PosEncoder, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        self.dropout_rate = pos_dropout_rate

        # TODO: nanishiteru?
        position = torch.arange(self.max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2)
            * (-math.log(10000.0) / self.embed_dim)
        )
        pe = torch.zeros(self.max_seq_len, 1, self.embed_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        pos_emb = self.pe[: x.size(0)]
        x = x + self.alpha * pos_emb
        x = self.dropout(x)

        return x
