import os
import re
import numpy as np
import pandas as pd

import nltk
from hanspell import spell_checker
from konlpy.tag import Okt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from pykospacing import Spacing
from soynlp.normalizer import *
from tqdm import tqdm

# nltk.download("punkt_tab")
# nltk.download("punkt")
nltk.download("stopwords")


def clean_and_normalize_text(text):
    # <PERSON> 토큰을 전처리 과정에서 보호
    text = re.sub(r"<PERSON>", "PERSONTOKEN", text)
    text = text.lower()
    text = re.sub(r"[^a-zㄱ-ㅎ가-힣0-9\s?!;]", "", text)
    # 보호했던 <PERSON> 토큰을 복원
    text = re.sub(r"persontoken", "<PERSON>", text)
    text = re.sub(r"[!?;]+", lambda m: m.group(0)[0], text)
    return text


# https://github.com/ssut/py-hanspell/issues/47#issue-2047956388
def spell_check(text):
    result = spell_checker.check(text)
    return result.checked


def remove_repeat_text(text):
    result = repeat_normalize(text, num_repeats=3)
    return result


def open_textfile(file_path):
    f = open(file_path, "r")
    data = f.read()
    f.close()
    return data


def remove_english_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    result = [word for word in word_tokens if word.lower() not in stop_words]
    return " ".join(result)


def remove_korean_stopwords(text):
    okt = Okt()  # https://konlpy.org/ko/latest/install/
    stop_words = open_textfile("./utils/korean_stopwords.txt")
    stop_words = set(word_tokenize(stop_words))
    word_tokens = okt.morphs(text)  # 형태소 분석
    result = [word for word in word_tokens if not word in stop_words]
    return " ".join(result)


def remove_stopwords(text, remove_english_stop=False, remove_korean_stop=False):
    if remove_english_stop:
        text = remove_english_stopwords(text)
    if remove_korean_stop:
        text = remove_korean_stopwords(text)
    return text


def preprocess_text(text):
    text = clean_and_normalize_text(text)
    # text = Spacing()(text)  # Uncomment if needed
    text = spell_check(text)
    text = remove_repeat_text(text)
    # text = remove_stopwords(text, remove_english_stop=False, remove_korean_stop=False)
    return text


def preprocess_data(df):
    tqdm.pandas()
    columns_to_preprocess = ["sentence_1", "sentence_2"]

    for column in columns_to_preprocess:
        tqdm.write(f"Processing column: {column}")
        df[column] = df[column].progress_apply(preprocess_text)

    return df


def apply_preprocess(train ,dev, data_dir, preprocess=False):
    preprocessed_train_dir = os.path.join(data_dir, "preprocessed_train.csv")
    preprocessed_dev_dir = os.path.join(data_dir, "preprocessed_dev.csv")
    if preprocess == True:
        if os.path.exists(preprocessed_train_dir) and os.path.exists(
            preprocessed_dev_dir
        ):
            print("Loading preprocessed data...")
            preprocessed_train = pd.read_csv(preprocessed_train_dir, dtype={"label": np.float32})
            preprocessed_dev = pd.read_csv(preprocessed_dev_dir, dtype={"label": np.float32})
        else:
            print("Preprocessing train data...")
            preprocessed_train = preprocess_data(train)
            print(f"Saving preprocessed train data to {preprocessed_train_dir}")
            preprocessed_train.to_csv(preprocessed_train_dir, index=False)
            print("Preprocessing dev data...")
            preprocessed_dev = preprocess_data(dev)
            print(f"Saving preprocessed dev data to {preprocessed_dev_dir}")
            preprocessed_dev.to_csv(preprocessed_dev_dir, index=False)
    return preprocessed_train, preprocessed_dev