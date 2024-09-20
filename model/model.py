import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from scipy.stats import pearsonr
from torchmetrics.functional import pearson_corrcoef
from transformers import AutoModel


class STSModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.mod = AutoModel.from_pretrained(config["MODEL_NAME"])
        self.dense = nn.Linear(384, 1)
        self.sigmoid = nn.Sigmoid()
        self.lr = config["LEARNING_RATE"]

    def forward(self, input_ids, attention_mask):
        outputs = self.mod(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.dense(outputs.last_hidden_state[:, 0, :])
        return self.sigmoid(outputs) * 5

    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"].squeeze(), batch["attention_mask"].squeeze())
        loss = nn.MSELoss()(outputs, batch["labels"])
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"].squeeze(), batch["attention_mask"].squeeze())
        loss = nn.MSELoss()(outputs, batch["labels"])
        pearson_corr = pearson_corrcoef(outputs, batch["labels"])
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_pearson_corr", pearson_corr, on_step=False, on_epoch=True)
        return {"val_loss": loss, "predictions": outputs, "targets": batch["labels"]}

    def test_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"].squeeze(), batch["attention_mask"].squeeze())
        return {"predictions": outputs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
