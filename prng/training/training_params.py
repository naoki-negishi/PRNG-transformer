# # Standard Library
# from dataclasses import dataclass
# from pydantic import BaseModel
#
#
# class ModelConfig(BaseModel):
#     name: str = "bert-base-cased"
#
#
# @dataclass
# class TrainConfig:
#     data_name: str = "yelp_review_full"
#     dryrun_epoch: int = 5
#     epoch: int = 20
#     train_batch_size: int = 16
#     valid_batch_size: int = 32
#     seed: int = 42
#     optimizer = {"name": "AdamW", "lr": 2.0e-5, "weight_decay": 0.01}
#     scheduler: str = "cosine_schedule_with_warmup"
#     loss: str = "CrossEntropyLoss"
#     metric: str = "accuracy"
#
#
# @dataclass
# class MyConfig:
#     model_cfg: ModelConfig = ModelConfig()
#     train_cfg: TrainConfig = TrainConfig()
#     output_dir: str = "../outputs"
#     device: str = "cuda"
#     dry_run: bool = True
