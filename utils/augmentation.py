import os
import random

import numpy as np
import pandas as pd
from konlpy.tag import Mecab


def augment_data(df):
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
