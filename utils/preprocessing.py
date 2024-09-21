import re

import nltk
from hanspell import spell_checker
from konlpy.tag import Okt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pykospacing import Spacing
from soynlp.normalizer import *

# nltk.download("punkt_tab")
# nltk.download("punkt")
nltk.download("stopwords")


def remove_punctuation_and_special_chars(text):
    text = text.lower()
    return re.sub(r"[^a-z0-9가-힣\s]", "", text)


# https://github.com/ssut/py-hanspell/issues/47#issue-2047956388
def spell_check(text):
    result = spell_checker.check(text)
    return result.checked


# 네 번 이상 반복되는 것만 2개로 줄여줌
def remove_repeat_text(text):
    result = repeat_normalize(text, num_repeats=2)
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


def preprocessing(df):
    columns_to_preprocess = ["sentence_1", "sentence_2"]
    df[columns_to_preprocess] = df[columns_to_preprocess].map(
        remove_punctuation_and_special_chars
    )
    # df[columns_to_preprocess] = df[columns_to_preprocess].map(Spacing())
    df[columns_to_preprocess] = df[columns_to_preprocess].map(spell_check)
    df[columns_to_preprocess] = df[columns_to_preprocess].map(remove_repeat_text)
    df[columns_to_preprocess] = df[columns_to_preprocess].map(
        lambda x: remove_stopwords(
            x, remove_english_stop=False, remove_korean_stop=True
        )
    )
    return df
