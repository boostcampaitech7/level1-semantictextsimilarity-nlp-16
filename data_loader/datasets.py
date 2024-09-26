import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    """_summary_
    train, validation 데이터셋 클래스

    문장 쌍과 레이블을 포함하고 있으며,
    문장을 토큰화하여 모델 입력에 적합한 형태로 반환
    Args:
        sentence_1 (pd.Series[str]): 입력 문장 1
        sentence_2 (pd.Series[str]): 입력 문장 2
        labels (pd.Series[float]): 입력 문장 쌍에 대한 레이블
        tokenizer: 문장을 토큰화하기 위한 토크나이저
        max_len (int): 최대 시퀀스 길이
        truncation (bool, optional): 토큰화 시 문장 절단 여부
    """
    def __init__(
        self, sentence_1, sentence_2, labels, tokenizer, max_len, truncation=True
    ):
        self.sentence_1 = sentence_1
        self.sentence_2 = sentence_2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.truncation = truncation

    def __len__(self):
        return len(self.sentence_1)

    def __getitem__(self, idx):
        sentence_1 = self.sentence_1[idx]
        sentence_2 = self.sentence_2[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            sentence_1,
            sentence_2,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=self.truncation,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )
        return {
            "sentence_pair": [sentence_1, sentence_2],
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "token_type_ids": encoding["token_type_ids"],
            "labels": torch.tensor([label]),
        }


class TestDataset(Dataset):
    """_summary_
    test 데이터셋 클래스

    문장 쌍을 포함하고 있으며,
    문장을 토큰화하여 모델 입력에 적합한 형태로 반환
    Args:
        sentence_1 (pd.Series[str]): 입력 문장 1
        sentence_2 (pd.Series[str]): 입력 문장 2
        tokenizer: 문장을 토큰화하기 위한 토크나이저
        max_len (int): 최대 시퀀스 길이
        truncation (bool, optional): 토큰화 시 문장 절단 여부
    """
    def __init__(self, sentence_1, sentence_2, tokenizer, max_len, truncation=True):
        self.sentence_1 = sentence_1
        self.sentence_2 = sentence_2
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.truncation = truncation

    def __len__(self):
        return len(self.sentence_1)

    def __getitem__(self, idx):
        sentence_1 = self.sentence_1[idx]
        sentence_2 = self.sentence_2[idx]

        encoding = self.tokenizer(
            sentence_1,
            sentence_2,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=self.truncation,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )
        return {
            "sentence_pair": [sentence_1, sentence_2],
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "token_type_ids": encoding["token_type_ids"],
        }
