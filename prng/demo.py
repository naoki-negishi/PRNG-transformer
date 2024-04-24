# Third Party Library
import hydra
from omegaconf import OmegaConf, DictConfig
import torch
from loguru import logger

# First Party Library
from prng.prng.model.prng_bert import PRNGBERTLM
from prng.prng.models.tokenizer import NumberTokenizer
from prng.prng.models.utils import PRNGConfig


def load_model(cfg: DictConfig, tokenizer: NumberTokenizer) -> PRNGBERTLM:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Model: {cfg.model.name}, device: {device}")

    model_config = PRNGConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=cfg.model.embed_dim,
        max_seq_len=cfg.model.max_seq_len,
        pos_dropout_rate=cfg.model.pos_dropout_rate,
        transformer_dropout_rate=cfg.model.transformer_dropout_rate,
        layer_norm_eps=cfg.model.layer_norm_eps,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
    )
    if cfg.model.name == "prng_bert":
        model = PRNGBERTLM(model_config).to(device)
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")

    return model


def preprocess_input_text(
    input_text: str, tokenizer: NumberTokenizer
) -> tuple[torch.Tensor, torch.Tensor]:
    pass


@logger.catch
@hydra.main(version_base=None, config_path="../configs/", config_name="demo")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    tokenizer = NumberTokenizer(vocab_size=cfg.model.vocab_size)
    model = load_model(cfg, tokenizer)

    while True:
        input_text = input("Enter a number sequence (enter 'q' to exit): \n")
        if input_text == "q":
            break

    input_ids, segment_label = preprocess_input_text(input_text, tokenizer)
    with torch.no_grad():
        enc_output = model(input_ids, segment_label)
        pred_num_seq = tokenizer.decode(enc_output)
        # TODO: revise
        masked_token = [t for t in pred_num_seq if t != tokenizer.mask_token]
        logger.info(masked_token)


if __name__ == "__main__":
    main()
