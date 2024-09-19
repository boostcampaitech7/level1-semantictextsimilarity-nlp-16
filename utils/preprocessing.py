"""
띄어쓰기, 맞춤법 교정: https://github.com/ssut/py-hanspell
정규화
- ㅋㅋㅋ등 중복 문자 처리: SOYNLP
- 줄임말, 신조어 체크: 토큰화를 이상하게 분해하는 문제, 신조어나 
"""

from hanspell import spell_checker
from soynlp.normalizer import *


def spell_check(text):
    result = spell_checker.check(text)
    return result.checked

def remove_repeat_text(text):
    return emoticon_normalize(text, num_repeats=2)

def preprocessing(data):
    for row in data:
        row['sentence_1'] = spell_check(row['sentence_1'])
        row['sentence_1'] = remove_repeat_text(row['sentence_1'])

        row['sentence_2'] = spell_check(row['sentence_2'])
        row['sentence_2'] = remove_repeat_text(row['sentence_2'])

    return data
