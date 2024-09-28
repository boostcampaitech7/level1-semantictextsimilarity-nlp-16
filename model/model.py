import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torchmetrics.functional import pearson_corrcoef
from transformers import AutoModel


class STSModel(L.LightningModule):
    """_summary_
    LoRA를 적용한 Semantic Textual Similarity (STS)

    pretrain된 언어모델에 LoRA를 적용하여 fine-tune
    문장 쌍을 모델 입력으로 받아 두 문장의 유사도를 도출
    Args:
        config (dict): model configuration
        model (nn.Module): pre-trained model
    Attributes:
        model (PeftModel): LoRA를 적용한 PEFT model
        dense (nn.Linear): last hidden state를 scalar로 projection하는 layer
        sigmoid (nn.Sigmoid): output을 0~1 사이의 값으로 반환
        loss (nn.MSELoss): Mean Squared Error loss
        lr (float): learning rate
    """
    def __init__(self, config, model):
        super().__init__()
        self.save_hyperparameters(config)

        peft_config = LoraConfig(
            r=config["LORA_RANK"],
            lora_alpha=(16**2) / config["LORA_RANK"],
            target_modules=config["MODULE_NAMES"],
            lora_dropout=0.05,
            bias="none",
        )

        self.mod = get_peft_model(model, peft_config)
        self.dense = nn.Linear(model.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()
        self.lr = config["LEARNING_RATE"]

    def forward(self, input_ids, attention_mask):
        """_summary_
        modeld의 forward pass를 정의

        Args:
            input_ids (torch.tensor): 토큰화된 input text
            attention_mask (torch.tensor): input_ids에 대해 padding index를 나타내는 attention mask

        Returns:
            _type_: 0 ~ 5 의 값을 갖는 label score
        """
        outputs = self.mod(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.dense(outputs.last_hidden_state[:, 0, :])
        return self.sigmoid(outputs) * 5

    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"].squeeze(), batch["attention_mask"].squeeze())
        loss = self.loss(outputs, batch["labels"])
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"].squeeze(), batch["attention_mask"].squeeze())
        loss = self.loss(outputs, batch["labels"])
        pearson_corr = pearson_corrcoef(outputs, batch["labels"])
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_pearson_corr", pearson_corr, on_step=False, on_epoch=True)
        return {"val_loss": loss, "predictions": outputs, "targets": batch["labels"]}

    def predict_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"].squeeze(), batch["attention_mask"].squeeze())
        return outputs.squeeze().tolist()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
