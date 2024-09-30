import os
import re

import nltk
import numpy as np
import pandas as pd
from hanspell import spell_checker
from konlpy.tag import Okt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pykospacing import Spacing
from soynlp.normalizer import *
from tqdm import tqdm

# nltk.download("punkt_tab")
# nltk.download("punkt")
# nltk.download("stopwords")


def clean_and_normalize_text(text):
    text = re.sub(r"<PERSON>", "PERSONTOKEN", text)
    text = text.lower()
    text = re.sub(r"[^a-zㄱ-ㅎ가-힣0-9\s?!;]", "", text)
    text = re.sub(r"persontoken", "<PERSON>", text)
    text = re.sub(r"[!?;]+", lambda m: m.group(0)[0], text)
    return text


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
    okt = Okt()
    stop_words = open_textfile("./utils/korean_stopwords.txt")
    stop_words = set(word_tokenize(stop_words))
    word_tokens = okt.morphs(text)
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
    # text = Spacing()(text)
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


def apply_preprocess(df, data_dir, name_for_save, preprocess=False):
    """_summary_
    dataframe에 preprocess를 적용

    data directory에 preprocessed data가 존재하는지 확인하고,
    없는 경우에만 preprocessing 적용 및 전처리된 data를 csv로 생성
    bool type의 preprocess argument를 통해 preprocessing 적용 여부 결정

    Args:
        df (pd.DataFrame): input dataset
        data_dir (str): data directory 경로
        name_for_save (str): data 저장 파일명
        preprocess (bool, optional): preprocess 적용 여부

    Returns:
        preprocessed_df (pd.DataFrame): preprocess 적용된 dataframe
    """
    preprocessed_df_dir = os.path.join(data_dir, name_for_save)
    if preprocess == True:
        if os.path.exists(preprocessed_df_dir):
            print(f"Loading {name_for_save}...")
            preprocessed_df = pd.read_csv(
                preprocessed_df_dir, dtype={"label": np.float32}
            )
        else:
            print("Preprocessing data...")
            preprocessed_df = preprocess_data(df)
            print(f"Saving preprocessed data to {preprocessed_df_dir}")
            preprocessed_df.to_csv(preprocessed_df_dir, index=False)
        return preprocessed_df
    else:
        return df
