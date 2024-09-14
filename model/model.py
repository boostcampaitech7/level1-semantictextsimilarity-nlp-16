import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import pearson_corrcoef
import pytorch_lightning as pl
from transformers import AutoModel
from scipy.stats import pearsonr
import wandb


class STSModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.mod = AutoModel.from_pretrained(config['MODEL_NAME'])
        self.dense = nn.Linear(384, 1)
        self.sigmoid = nn.Sigmoid()
        self.lr = config['LEARNING_RATE']
        
    def forward(self, input_ids, attention_mask):
        outputs = self.mod(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.dense(outputs.last_hidden_state[:, 0, :])
        return self.sigmoid(outputs) * 5
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch['input_ids'].squeeze(), batch['attention_mask'].squeeze())
        loss = nn.MSELoss()(outputs, batch['labels'])
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch['input_ids'].squeeze(), batch['attention_mask'].squeeze())
        loss = nn.MSELoss()(outputs, batch['labels'])
        pearson_corr = pearson_corrcoef(outputs, batch['labels'])
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_pearson_corr', pearson_corr, on_step=False, on_epoch=True)
        return {'val_loss': loss, 'predictions': outputs, 'targets': batch['labels']}
    
    def test_step(self, batch, batch_idx):
        emb_sen1 = self(batch['input_ids'][0].squeeze(), batch['attention_mask'][0].squeeze())
        emb_sen2 = self(batch['input_ids'][1].squeeze(), batch['attention_mask'][1].squeeze())
        similarity = self.cosine_similarity(emb_sen1, emb_sen2)
        return {'predictions' : similarity}
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)