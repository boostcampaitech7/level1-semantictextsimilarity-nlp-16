import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel
from scipy.stats import pearsonr
import wandb


class STSModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.mod = AutoModel.from_pretrained(config['MODEL_NAME'])
        self.mod.train()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        self.lr = config['LEARNING_RATE']
        
    def forward(self, input_ids, attention_mask):
        outputs = self.mod(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]
    
    def training_step(self, batch, batch_idx):
        emb_sen1 = self(batch['input_ids'][0].squeeze(), batch['attention_mask'][0].squeeze())
        emb_sen2 = self(batch['input_ids'][1].squeeze(), batch['attention_mask'][1].squeeze())
        similarity = self.cosine_similarity(emb_sen1, emb_sen2)
        similarity = 2.5*similarity + 2.5
        loss = nn.MSELoss()(similarity, batch['labels'].squeeze())
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        emb_sen1 = self(batch['input_ids'][0].squeeze(), batch['attention_mask'][0].squeeze())
        emb_sen2 = self(batch['input_ids'][1].squeeze(), batch['attention_mask'][1].squeeze())
        similarity = self.cosine_similarity(emb_sen1, emb_sen2)
        similarity = 2.5*similarity + 2.5
        loss = nn.MSELoss()(similarity, batch['labels'].squeeze())
        self.log('val_loss', loss)
        return {'val_loss': loss, 'predictions': similarity, 'targets': batch['labels']}
    
    def on_validation_epoch_end(self, outputs):
        predictions = torch.cat([x['predictions'] for x in outputs]).cpu().numpy()
        targets = torch.cat([x['targets'] for x in outputs]).cpu().numpy()
        pearson_corr, _ = pearsonr(targets, predictions)
        self.log('val_pearson_corr', pearson_corr)
    
    def test_step(self, batch, batch_idx):
        emb_sen1 = self(batch['input_ids'][0].squeeze(), batch['attention_mask'][0].squeeze())
        emb_sen2 = self(batch['input_ids'][1].squeeze(), batch['attention_mask'][1].squeeze())
        similarity = self.cosine_similarity(emb_sen1, emb_sen2)
        return {'predictions' : similarity}
    
    def test_epoch_end(self, outputs):
        all_predictions = torch.cat([x['predictions'] for x in outputs]).cpu().numpy()
        np.savetxt('test_predictions.txt', all_predictions)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)