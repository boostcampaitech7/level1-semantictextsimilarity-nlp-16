import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .datasets import TestDataset, TextDataset


class TextDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer,
        max_len,
        train_data=None,
        dev_data=None,
        test_data=None,
        truncation=True,
        batch_size=32,
    ):
        super().__init__()
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.truncation = truncation
        self.batch_size = batch_size

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = TextDataset(
                sentence_1=self.train_data["sentence_1"],
                sentence_2=self.train_data["sentence_2"],
                labels=self.train_data["label"],
                tokenizer=self.tokenizer,
                max_len=self.max_len,
            )
            self.dev_dataset = TextDataset(
                sentence_1=self.dev_data["sentence_1"],
                sentence_2=self.dev_data["sentence_2"],
                labels=self.dev_data["label"],
                tokenizer=self.tokenizer,
                max_len=self.max_len,
            )
        else:
            self.test_dataset = TestDataset(
                sentence_1=self.test_data["sentence_1"],
                sentence_2=self.test_data["sentence_2"],
                tokenizer=self.tokenizer,
                max_len=self.max_len,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
