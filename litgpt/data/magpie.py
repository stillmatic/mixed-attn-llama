# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import List, Optional, Union
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from litgpt import Tokenizer
from litgpt.data import DataModule, SFTDataset, get_sft_collate_fn
from litgpt.prompts import PromptStyle

DATASET_NAME = "Magpie-Align/Magpie-Pro-300K-Filtered"


@dataclass
class Magpie(DataModule):
    """The Magpie data module for SFT."""

    mask_prompt: bool = False
    prompt_style: Union[str, PromptStyle] = "alpaca"
    ignore_index: int = -100
    """The path to the data directory, containing two folders 'train' and 'val'
    which are the output of the preprocessing step. The path can also be a remote path (e.g., s3://)."""
    data_path: Union[str, Path] = Path("data/magpie")
    val_split_fraction: float = 0.0005
    """The fraction of data that should be put aside for validation."""
    seed: int = 42
    """The seed to use for shuffling the training data."""
    num_workers: int = 8
    """The number of workers to use for the dataloaders."""

    tokenizer: Optional[Tokenizer] = field(default=None, repr=False, init=False)
    batch_size: int = field(default=1, repr=False, init=False)
    seq_length: int = field(default=2048, repr=False, init=False)
    max_seq_length: int = field(default=-1, init=False, repr=False)
    
    def __post_init__(self) -> None:
        # Could be a remote path (s3://) or a local path
        self.data_path_train = str(self.data_path).rstrip("/") + "/train"
        self.data_path_val = str(self.data_path).rstrip("/") + "/val"

    def connect(
        self,
        tokenizer: Optional[Tokenizer] = None,
        batch_size: int = 1,
        max_seq_length: Optional[int] = 2048,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = (
            max_seq_length + 1
        )  # Increase by one because we need the next token as well

    def prepare_data(self) -> None:
        from datasets import Dataset, load_dataset
        from litdata import optimize

        if str(self.data_path).startswith("s3://"):
            print(
                f"The OpenWebText data path points to an S3 location: {self.data_path}. Skipping preprocessing."
            )
            return

        if Path(self.data_path_train).is_dir() and Path(self.data_path_val).is_dir():
            print(
                f"Found OpenWebText train and val dir: {self.data_path}. Skipping preprocessing."
            )
            return

        dataset = load_dataset(
            DATASET_NAME, num_proc=(os.cpu_count() // 2), trust_remote_code=True
        )

        # Split the data in training and validation
        split_dataset = dataset["train"].train_test_split(
            test_size=self.val_split_fraction, seed=self.seed, shuffle=True
        )
        split_dataset["val"] = split_dataset.pop("test")  # rename the test split to val

        train_data = format_dataset(split_dataset["train"])
        test_data = format_dataset(split_dataset["val"])

        self.train_dataset = SFTDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )
        self.test_dataset = SFTDataset(
            data=test_data,
            tokenizer=self.tokenizer,
            prompt_style=self.prompt_style,
            max_seq_length=self.max_seq_length,
            mask_prompt=self.mask_prompt,
            ignore_index=self.ignore_index,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(
                max_seq_length=self.max_seq_length, ignore_index=self.ignore_index
            ),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=get_sft_collate_fn(
                max_seq_length=self.max_seq_length, ignore_index=self.ignore_index
            ),
        )

def format_dataset(dataset: List[dict], include_multi_turn_conversations: bool = False) -> List[dict]:
    formatted = []

    for entry in tqdm(dataset):
        convo = entry["conversations"]
        if include_multi_turn_conversations:
            for i in range(0, len(convo) - 1, 2):
                formatted.append({"instruction": convo[i]["content"], "input": "", "output": convo[i + 1]["content"]})
        else:
            formatted.append({"instruction": convo[0]["value"], "input": "", "output": convo[1]["value"]})

    return formatted
