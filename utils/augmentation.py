import os
import random

import numpy as np
import pandas as pd
from konlpy.tag import Mecab


def augment_data(df):
    """_summary_
    Sentence swap augmentation 정의

    Args:
        df (pd.DataFrame): augmentation을 적용할 dataframe

    Returns:
        combined_df (pd.DataFrame): augmentation이 적용된 dataframe
    """
    augmented_df = df.copy()
    ## 문장 위치 변경
    augmented_df["sentence_1"], augmented_df["sentence_2"] = (
        augmented_df["sentence_2"],
        augmented_df["sentence_1"],
    )
    ## 원본 데이터와 증강된 데이터 합치기
    combined_df = pd.concat([df, augmented_df], axis=0)
    return combined_df.reset_index(drop=True)


def apply_augment(train, data_dir, augment=False):
    """_summary_
    dataframe에 augmentation 적용

    data directory에 augmented된 data가 존재하는지 확인하고,
    없는 경우에만 augementation 적용 및 augmented_trian.csv 생성
    bool type의 augment argument를 통해 augmentation 적용 여부 결정
    Args:
        train (pd.DataFrame): train dataset
        data_dir (str): data directory 경로
        augment (bool, optional): augmentation 적용 여부

    Returns:
        augmented_train (pd.DataFrame): augmentation 적용된 dataframe
    """
    augmented_train_dir = os.path.join(data_dir, "augmented_train.csv")
    augmented_dev_dir = os.path.join(data_dir, "augmented_dev.csv")
    if augment:
        if os.path.exists(augmented_train_dir):
            print("Loading augmented data...")
            augmented_train = pd.read_csv(
                augmented_train_dir, dtype={"label": np.float32}
            )
        else:
            print("Augmenting train data...")
            augmented_train = augment_data(train)
            print(f"Saving augmented train data to {augmented_train_dir}")
            augmented_train.to_csv(augmented_train_dir, index=False)
    return augmented_train


def random_deletion(text, p=0.2):
    """_summary_
    임의 토큰 삭제를 통한 augmentation

    형태소 분석을 활용한 한국어 토큰화 및
    감탄사, 조사, 어미, 접두사, 접미사에 해당하는 토큰만 제거
    제거한 토큰 수에 비례하여 label score 차감
    Args:
        text (str): augmentation 적용할 text
        p (float, optional): 문장 내에서 삭제할 토큰의 비율

    Returns:
        str: 토큰이 임의로 삭제된 text
    """
    mecab = Mecab()
    tokens_with_pos = mecab.pos(text)
    tokens = [token for token, _ in tokens_with_pos]
    target_tags = ["IC", "J", "E", "XP", "XS"]

    if len(tokens) == 1:
        return text

    remaining = [
        (token, pos)
        for token, pos in tokens_with_pos
        if random.random() > p or not any([pos.startswith(tag) for tag in target_tags])
    ]

    if len(remaining) == 0:
        return random.choice(tokens)

    result = []
    for token, pos in remaining:
        if pos.startswith("J") or pos.startswith("E"):
            if len(result) != 0:
                result[-1] += token
            else:
                result.append(token)
        else:
            result.append(token)

    return " ".join(result)


def apply_random_deletion(train):
    """_summary_
    dataframe에 random deletion을 적용

    Args:
        train (pd.DataFrame): train dataset

    Returns:
        pd.DataFrame: augmentation 적용된 dataframe
    """
    train_deleted = []
    for _, row in train.iterrows():
        deleted_text = random_deletion(row["sentence_1"])
        diff = 0.2 * (len(row["sentence_1"]) - len(deleted_text))
        row["sentence_1"] = deleted_text
        row["label"] = max(row["label"] - diff, 0)
        train_deleted.append(row)

    train_deleted = pd.DataFrame(train_deleted)
    train = pd.concat([train, train_deleted]).drop_duplicates()

    return train.reset_index(drop=True, inplace=True)
