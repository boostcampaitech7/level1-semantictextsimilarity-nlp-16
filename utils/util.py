import random

import numpy as np
import torch

import wandb


def model_load(run_name, model_path, project_name="Level1-STS"):
    run = wandb.init(project=project_name, name=run_name)
    config = run.config

    artifact = wandb.use_artifact(f"{project_name}/{model_path}:latest")
    model_dir = artifact.download(root="./saved")

    model = torch.load(f"{model_dir}/model.pth")

    return model, config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
