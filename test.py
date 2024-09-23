import os
import argparse

import pandas as pd
import wandb
from pytorch_lightning import Trainer

from data_loader.data_loaders import TextDataLoader
from model.model import STSModel
from utils.preprocessing import preprocessing
from utils.tokenizer import get_tokenizer
from utils.clean import clean_texts


def main(arg):
    ## data reading
    test_dir = os.path.join(arg.data_dir, 'test.csv')
    submission_dir = os.path.join(arg.data_dir, 'sample_submission.csv')

    test = pd.read_csv(test_dir)
    submission = pd.read_csv(submission_dir)

    ## model/config loading
    wandb.login()

    api = wandb.Api()
    run = api.run(arg.run_path)

    artifact = api.artifact(arg.model_path)
    model_dir = artifact.download()
    config = run.config

    model = STSModel.load_from_checkpoint(f'{model_dir}/model.ckpt')

    ## processing
    test['sentence_1'] = clean_texts(test['sentence_1'])
    test['sentence_2'] = clean_texts(test['sentence_2'])

    test = test.dropna(subset=['sentence_1', 'sentence_2'])
    test = test.reset_index(drop=True)

    tokenizer = get_tokenizer(config['MODEL_NAME'])
    dataloader = TextDataLoader(
        tokenizer=tokenizer,
        max_len=config['MAX_LEN'],
        test_data=test,
        truncation=True
    )
        
    trainer = Trainer(
        accelerator="gpu",
        devices=1
    )

    preds = trainer.predict(model, dataloader)
    all_pred = [val for pred in preds for val in pred]
    submission['target'] = all_pred

    print(submission.head())
    submission.to_csv('data/submission.csv', index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-d",
        "--data_dir",
        default=None,
        type=str,
        help="directory path for data (default: None)",
    )
    args.add_argument(
        "-r",
        "--run_path",
        default=None,
        type=str,
        help="wandb run path for an experiment (default: None)",
    )
    args.add_argument(
        "-m",
        "--model_path",
        default=None,
        type=str,
        help="artifact path for a model (default: all)",
    )

    arg = args.parse_args()
    main(arg)
