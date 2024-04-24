# Standard Library
import math
import random
import sys
from datetime import datetime
from pathlib import Path

# Third Party Library
import hydra
import numpy as np
import torch
import wandb
from loguru import logger
from omegaconf import OmegaConf, DictConfig
from torch import nn
from transformers import (
    EvalPrediction,
    Trainer,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
)

# First Party Library
from prng.prng.models.prng_bert import PRNGBERTLM
from prng.prng.models.utils import NumberTokenizer, PRNGConfig
from prng.prng.preprocess.datasets import PRNGDataCollator, PRNGDataset

sys.path.append("..")


def set_seed_all(seed=42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True


def wandb_setting(cfg) -> None:
    wandb.login()
    config = {
        "model_name": cfg.model.name,
        "data_name": cfg.data.name,
        "epoch": cfg.train.epoch,
        "train_batch_size": cfg.train.train_batch_size,
        "valid_batch_size": cfg.train.valid_batch_size,
        "seed": cfg.train.seed,
        "optimizer": cfg.train.optimizer,
        "scheduler": cfg.train.scheduler,
        "loss": cfg.train.loss,
        "metric": cfg.train.metric,
    }
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name
        + "_"
        + cfg.model.name
        + "_"
        + str(datetime.now().strftime("%Y%m%d%H%M")),
        config=config,
    )

    def make_optimizer(params, name, **kwargs):
        optimizer = torch.optim.__dict__[name](params, **kwargs)
        return optimizer


def make_optimizer(params, name, **kwargs) -> torch.optim.Optimizer:
    optimizer = torch.optim.__dict__[name](params, **kwargs)
    return optimizer


def compute_accuracy(eval_pred: EvalPrediction) -> dict[str, float]:
    predictions = eval_pred.predictions[0]  # from ModelOutput.logits
    predictions = np.argmax(predictions, axis=-1)
    labels = eval_pred.label_ids  # from label_names=["segment_labels"]

    batch_size = predictions.shape[0]
    seq_len = predictions.shape[1]
    assert predictions.shape == labels.shape == (batch_size, seq_len)

    return {"accuracy": (predictions == labels).mean()}


def training(cfg: DictConfig) -> None:
    set_seed_all(cfg.train.seed)
    wandb_setting(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # define tokenizer
    tokenizer = NumberTokenizer(max_num=cfg.data.maximum_number)
    # load dataset
    train_dataset = PRNGDataset(cfg.data.train, tokenizer=tokenizer)
    valid_dataset = PRNGDataset(cfg.data.valid, tokenizer=tokenizer)
    # define tokenizer and data_collator
    data_collator = PRNGDataCollator(tokenizer=tokenizer)

    # define model
    logger.info(f"Loading Model: {cfg.model.name}")
    model_config = PRNGConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=cfg.model.embed_dim,
        max_seq_len=cfg.data.max_seq_len,
        pos_dropout_rate=cfg.model.pos_dropout_rate,
        transformer_dropout_rate=cfg.model.transformer_dropout_rate,
        layer_norm_eps=cfg.model.layer_norm_eps,
        num_hidden_layers=cfg.model.num_hidden_layers,
        num_attention_heads=cfg.model.num_attention_heads,
        padding_idx=tokenizer.pad_token_id,
    )
    if cfg.model.name == "prng_bert":
        # TODO: Does the loss function behave as expected?
        loss_func = nn.NLLLoss(ignore_index=model_config.padding_idx)
        model = PRNGBERTLM(model_config, loss_func).to(device)
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")

    # define optimizer and scheduler
    params = filter(lambda x: x.requires_grad, model.parameters())
    # If you want to rewrite the optimizer, please rewrite it directly
    optimizer = make_optimizer(params, **cfg["train"]["optimizer"])
    num_warmup_steps = (
        math.ceil(len(train_dataset) / cfg.train.train_batch_size) * 1
    )
    num_training_steps = (
        math.ceil(len(valid_dataset) / cfg.train.train_batch_size) * 10
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # define training arguments
    logger.info(f"Training: {cfg.train.epoch} epochs")
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.train.train_batch_size,
        per_device_eval_batch_size=cfg.train.valid_batch_size,
        num_train_epochs=cfg.train.epoch,
        save_strategy="epoch",
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        label_names=["segment_labels"],
        metric_for_best_model=cfg.train.metric,
        report_to="wandb",
    )
    # run training
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_accuracy,
        optimizers=[optimizer, scheduler],
    )
    trainer.train()


@logger.catch
@hydra.main(version_base=None, config_path="../../../prng/configs/", config_name="test")
def main(cfg: DictConfig) -> None:
    # Setup logger
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    log_file = cfg.output_dir + "/training_{time}.log"
    logger.add(log_file)
    logger.info(OmegaConf.to_yaml(cfg))

    # training
    training(cfg)
    logger.info("Training all done!")


if __name__ == "__main__":
    main()
