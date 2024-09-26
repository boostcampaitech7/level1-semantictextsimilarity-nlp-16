import argparse
import glob
import os

import numpy as np
import pandas as pd


def main(arg):
    path = os.path.join(arg.data_dir, "submission")
    all_files = glob.glob(os.path.join(path, "*.csv"))

    # 각 submission 파일에서 'target' 열을 가져옴
    preds = [pd.read_csv(file)[["target"]] for file in all_files]

    # 가중치 설정 (예를 들어, 성능에 따라 가중치 부여)
    w = [30, 30, 15]

    # 가중치 앙상블
    ensemble_preds = np.average(preds, axis=0, weights=w)

    # 결과 저장
    submission = pd.read_csv(f"{arg.data_dir}/sample_submission.csv")
    submission["target"] = ensemble_preds
    submission.to_csv(f"{path}/ensemble/weighted_voting.csv", index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "-d",
        "--data_dir",
        default=None,
        type=str,
        help="directory path for data (default: None)",
    )

    arg = args.parse_args()
    main(arg)
