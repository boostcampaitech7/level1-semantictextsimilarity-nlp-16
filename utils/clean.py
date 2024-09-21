import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from konlpy.tag import Okt

nltk.download("punkt_tab")
nltk.download("punkt")
nltk.download("stopwords")


def remove_punctuation_and_special_chars(text):
    return re.sub(r"[^a-z0-9가-힣\s]", "", text)

def remove_english_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    result = [word for word in word_tokens if word.lower() not in stop_words]
    return " ".join(result)

def open_textfile(file_path):
    f = open(file_path, 'r')
    data = f.read()
    print(data)
    f.close()
    return data

def remove_korean_stopwords(text):
    okt = Okt() # https://konlpy.org/ko/latest/install/

    stop_words = open_textfile("./korean_stopwords.txt")
    print(stop_words)

    stop_words = set(word_tokenize(text))
    word_tokens = okt.morphs(text) # 형태소 단위로 분해
    print(stop_words)
    print(word_tokens)

    result = [word for word in word_tokens if not word in stop_words]
    return " ".join(result)

def clean_text(text, remove_english_stop=False, remove_korean_stop=False):
    text = text.lower()
    text = remove_punctuation_and_special_chars(text)

    if remove_english_stop:
        text = remove_english_stopwords(text)

    if remove_korean_stop:
        text = remove_korean_stopwords(text)
    return text

# def clean_text(text, remove_stop=False, language="english"):
#     text = text.lower()
#     text = remove_punctuation_and_special_chars(text)

#     if remove_stop:
#         text = remove_stopwords(text, language)
#     return text


# def clean_texts(texts, remove_stop=False, language="english"):
#     return [clean_text(text, remove_stop, language) for text in texts]
