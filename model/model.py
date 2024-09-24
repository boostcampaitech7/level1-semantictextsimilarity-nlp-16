import numpy as np
import pytorch_lightning as L
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torchmetrics.functional import pearson_corrcoef
from transformers import AutoModel


class STSModel(L.LightningModule):
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
