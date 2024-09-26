import pytorch_lightning as L
from torch.utils.data import DataLoader

from .datasets import TestDataset, TrainDataset


class TextDataLoader(L.LightningDataModule):
    """_summary_
    텍스트 데이터를 처리 및 로드

    train, validation, predict 데이터셋을 관리하고,
    분석의 각 단계에 맞는 데이터로더를 제공
    Args:
        tokenizer: 문장을 토큰화하는 토크나이저
        max_len (int): 최대 시퀀스 길이
        train_data (pd.DataFrame): 학습 데이터
        val_data (pd.DataFrame): 검증 데이터
        predict_data (pd.DataFrame): 추론 대상 데이터
        truncation (bool, optional): 문장 절단 여부
        batch_size (int): 배치 크기
    """

    def __init__(
        self,
        tokenizer,
        max_len,
        train_data=None,
        val_data=None,
        predict_data=None,
        truncation=True,
        batch_size=32,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.train_data = train_data
        self.val_data = val_data
        self.predict_data = predict_data
        self.truncation = truncation
        self.batch_size = batch_size

    def setup(self, stage=None):
        """_summary_
        stage별 load할 데이터 정의

        Args:
            stage (str, optional): 'fit', 'test', 'predict', 기본값은 None
        """
        if stage == "fit":
            self.train_dataset = TrainDataset(
                sentence_1=self.train_data["sentence_1"],
                sentence_2=self.train_data["sentence_2"],
                labels=self.train_data["label"],
                tokenizer=self.tokenizer,
                max_len=self.max_len,
            )
            self.val_dataset = TrainDataset(
                sentence_1=self.val_data["sentence_1"],
                sentence_2=self.val_data["sentence_2"],
                labels=self.val_data["label"],
                tokenizer=self.tokenizer,
                max_len=self.max_len,
            )
        elif stage == "predict":
            self.predict_dataset = TestDataset(
                sentence_1=self.predict_data["sentence_1"],
                sentence_2=self.predict_data["sentence_2"],
                tokenizer=self.tokenizer,
                max_len=self.max_len,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=0)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=self.batch_size)
