import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, sentence_1, sentence_2, tokenizer, max_len, labels = None, truncation=True):
        self.sentence_1 = sentence_1
        self.sentence_2 = sentence_2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.truncation = truncation

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sentence_1 = self.sentence_1[idx]
        sentence_2 = self.sentence_2[idx]
        if self.labels is not None:
            label = self.labels[idx]

        encoding = self.tokenizer(
            sentence_1,
            sentence_2,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=self.truncation,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt',
        )

        return {
            'sentence_pair': [sentence_1, sentence_2],
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'token_type_ids': encoding['token_type_ids'],
            'labels': torch.tensor([label]) if self.labels is not None else None
        }
