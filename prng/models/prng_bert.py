# Standard Library
import math

# Third Party Library
import torch
from torch import nn
from transformers.modeling_outputs import ModelOutput

# First Party Library
from prng.prng.models.utils import PosEncoder, PRNGConfig


class PRNGBERT(nn.Module):
    def __init__(self, config: PRNGConfig) -> None:
        """
        Args:
            config: PRNGConfig, configuration for PRNGBERT
        """
        super(PRNGBERT, self).__init__()
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.max_seq_len = config.max_seq_len
        self.transformer_dropout_rate = config.transformer_dropout_rate
        self.layer_norm_eps = config.layer_norm_eps
        self.n_layers = config.num_hidden_layers
        self.attn_heads = config.num_attention_heads
        self.padding_idx = config.padding_idx

        self.embed = nn.Embedding(
            self.vocab_size, self.embed_dim, padding_idx=0
        )
        self.pos_encoder = PosEncoder(
            self.max_seq_len, self.embed_dim, config.pos_dropout_rate
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.attn_heads,
            dim_feedforward=self.embed_dim * 4,  # from BERT paper
            dropout=self.transformer_dropout_rate,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor
                 with shape (batch_size, seq_len)
        Returns:
            last_hidden (torch.Tensor): transformer encoded output
                 with shape (batch_size, seq_len , embedding_dim)
        """
        # Embedding and positional encoding
        emb_output = self.embed(x) * math.sqrt(self.embed_dim)  # TODO: sqrt?
        pos_emb_output = self.pos_encoder(emb_output)
        assert pos_emb_output.size() == torch.Size([x.size(0), x.size(1), self.embed_dim])

        # Attention (mask padding tokens with shape (batch_size, seq_len))
        attention_mask = x.eq(self.padding_idx)
        last_hidden = self.transformer_encoder(
            torch.permute(pos_emb_output, (1, 0, 2)),
            src_key_padding_mask=attention_mask
        )
        last_hidden = torch.permute(last_hidden, (1, 0, 2))
        assert last_hidden.size() == torch.Size([x.size(0), x.size(1), self.embed_dim])

        return last_hidden


# Masked LM
class PRNGBERTLM(nn.Module):
    def __init__(self, config: PRNGConfig, loss_func: torch.nn.Module) -> None:
        super(PRNGBERTLM, self).__init__()
        self.bert = PRNGBERT(config)
        # self.nsp_linear = nn.Linear(config.embed_dim, 2)
        self.mlm_linear = nn.Linear(config.embed_dim, config.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.loss_func = loss_func  # nn.NLLLoss(ignore_index=padding_idx)

        # TODO: is this affect the entire model?
        self.apply(self._init_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_labels: torch.Tensor
    ) -> ModelOutput:
        last_hidden = self.bert(input_ids)

        # Next sentence prediction
        # nsp_output = self.nsp_linear(last_hidden[:, 0])
        # nsp_output = self.softmax(nsp_output)

        # Masked language model
        mlm_output = self.mlm_linear(last_hidden)
        mlm_output = self.softmax(mlm_output)

        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)
        vocab_size = self.bert.vocab_size
        assert mlm_output.size() == torch.Size([batch_size, seq_len, vocab_size])
        assert segment_labels.size() == torch.Size([batch_size, seq_len])

        loss = self.loss_func(
            torch.permute(mlm_output, (0, 2, 1)),
            segment_labels
        )
        return ModelOutput(
            loss=loss,
            logits=mlm_output,
            last_hidden_state=last_hidden,
        )

    def _init_weights(self, module):
        # TODO: dropout, transformer's k, q, v, etc.
        # TODO: best inital value?
        # self.embedding.weight.data.uniform_(-initrange, initrange)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# Seq2Seq
class PRNGTransformer(nn.Module):
    def __init__(self, config: PRNGConfig):
        self.bert = PRNGBERT(config)

        self.apply(self._init_weights)

    def forward(self, x):
        pass

    def _init_weights(self, module):
        pass


# Causal LM
class PRNGGPT(nn.Module):
    def __init__(self, config: PRNGConfig):
        self.bert = PRNGBERT(config)

        self.apply(self._init_weights)

    def forward(self, x):
        # src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
        pass

    def _init_weights(self, module):
        pass
