# Standard Library
import json
import random

# Third Party Library
import torch
from tqdm import tqdm
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# First Party Library
from prng.prng.models.utils import NumberTokenizer


class PRNGDataset(Dataset):
    """ Contain number sequence data and mask tokens when creating DataLoader
    Args:
        file_path (str): path to the dataset file
    Attributes:
        tokenizer (NumberTokenizer): tokenizer for PRNG
        data (list[list[int]]): number sequence data
    """
    def __init__(self, file_path: str, tokenizer: NumberTokenizer) -> None:
        self.tokenizer = tokenizer
        self.data: list[list[int]] = []

        logger.info(f"Loading data from {file_path}")
        with open(file_path, "r") as f:
            for jsonl in tqdm(f):
                instance = json.loads(jsonl)
                num_seq: list[int] = instance["num_seq"]
                self.data.append(num_seq)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        """ Called when creating DataLoader, return masked number sequence
        Args:
            idx (int): index of the data
        Returns:
            num_seq (list[int]): randomly masked number sequence
            segment_label (list[int]): label for *masked* tokens (others are ignored label)
        """
        num_seq = self.data[idx]
        tokenized_num_seq = self.tokenizer.tokenize(num_seq)
        input_ids, segment_label = self.mask_tokens(tokenized_num_seq)
        return input_ids, segment_label

    def mask_tokens(self, num_seq: list[int]) -> tuple[list[int], list[int]]:
        seg_label = []  # for gold of randomly changed token
        for i, num in enumerate(num_seq):
            prob = random.random()
            # 15% randomly change token to mask token
            if prob < 0.15:
                seg_label.append(i)
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    num_seq[i] = self.tokenizer.mask_token_id

                # 10% randomly change token to random token
                elif prob < 0.9:
                    # special_token_num(=4) <= randint <= vocab_size-1 (=max_num + 4)
                    num_seq[i] = random.randint(
                        self.tokenizer.special_token_num,
                        self.tokenizer.vocab_size - 1,
                    )

                # 10% randomly change token to current token
                else:
                    num_seq[i] = num
            else:
                seg_label.append(self.tokenizer.pad_token_id)  # ignore label
                num_seq[i] = num

        assert len(num_seq) == len(seg_label)
        assert max(num_seq) < self.tokenizer.vocab_size, f"num_seq({self.tokenizer.vocab_size}<{max(num_seq)}): {num_seq}"
        return num_seq, seg_label


class PRNGDataCollator:
    """ Data collator that will dynamically pad the inputs received.
    """

    def __init__(self, tokenizer: NumberTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, instances: list[list[int]]) -> dict[str, torch.long]:
        num_seq, segment_labels = self.batching(instances)
        batch = {"input_ids": num_seq, "segment_labels": segment_labels}
        return batch

    def batching(
        self,
        instances: list[tuple[list[int], list[int]]],
    ) -> tuple[torch.long, torch.long]:
        """ Pad the input_ids and segment_label to the same length
        Args:
            instances (list[tuple[list[int], list[int]]):
                list of number sequence and segment label
        Returns:
            padded_input_ids (torch.long):
                padded input_ids with shape (batch_size, max_seq_len + 2)
            segment_labels (torch.long):
                padded segment_labels with shape (batch_size, max_seq_len + 2)
        """
        batch_input_ids: list[torch.long] = []
        segment_labels: list[torch.long] = []
        for input_ids, seg_labels in instances:
            # input_ids: [CLS] + shifted num_seq + [SEP]
            input_ids = torch.tensor(
                [self.tokenizer.cls_token_id]
                + input_ids
                + [self.tokenizer.sep_token_id]
            , dtype=torch.long)
            seg_labels = torch.tensor(
                [self.tokenizer.pad_token_id]
                + seg_labels
                + [self.tokenizer.pad_token_id]
            , dtype=torch.long)

            batch_input_ids.append(input_ids)
            segment_labels.append(seg_labels)
        assert len(batch_input_ids) == len(segment_labels)

        padded_input_ids = pad_sequence(
            batch_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        segment_labels = pad_sequence(
            segment_labels,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        batch_size = len(batch_input_ids)
        max_seq_len = padded_input_ids.size(1)
        assert padded_input_ids.size() == torch.Size([batch_size, max_seq_len])
        assert padded_input_ids.size(0) == segment_labels.size(0)

        return padded_input_ids, segment_labels
