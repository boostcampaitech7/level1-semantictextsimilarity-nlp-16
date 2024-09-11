import json
import torch
import random
import numpy as np
import pandas as pd
import wandb
import pytorch_lightning as pl
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
    
class WandbCheckpointCallback(pl.Callback):
    def __init__(self, top_k=3):
        self.top_k = top_k
        self.best_k_models = {}
        self.metric_score = None

    def on_validation_end(self, trainer, pl_module):
        # 현재 체크포인트의 성능 점수 가져오기
        metric_score = trainer.callback_metrics.get("val_loss")
        if metric_score:
            self.metric_score = metric_score.item()

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.metric_score is not None:
            # Wandb에 체크포인트 업로드
            artifact = wandb.Artifact(f"model-{trainer.current_epoch}", type="model")
            artifact.add_file(trainer.checkpoint_callback.best_model_path)
            trainer.logger.experiment.log_artifact(artifact)

            # 최상위 k개 모델 관리
            if len(self.best_k_models) < self.top_k:
                self.best_k_models[self.metric_score] = artifact
            else:
                worst_score = max(self.best_k_models.keys())
                if self.metric_score < worst_score:
                    # 가장 성능이 낮은 모델 삭제
                    artifact_to_delete = self.best_k_models.pop(worst_score)
                    trainer.logger.experiment.delete_artifact(artifact_to_delete.name)
                    # 새로운 모델 추가
                    self.best_k_models[self.metric_score] = artifact
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


