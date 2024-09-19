from hanspell import spell_checker
from soynlp.normalizer import *

"""
띄어쓰기: https://github.com/haven-jeon/PyKoSpacing
맞춤법 교정: https://github.com/ssut/py-hanspell
정규화
- ㅋㅋㅋ등 중복 문자 처리: https://github.com/lovit/soynlp
- 줄임말, 신조어 체크: soynlp 사용하여 처리 고민
"""

def spacing(text): # 미구현
    return text

def spell_check(text): # py-hanspell 패키지 버그 존재, 다른 패키지 찾아보기
    result = spell_checker.check(text)
    return result.checked

def remove_repeat_text(text):
    result = repeat_normalize(text) # 네번 이상 반복되는 것만 2개로 줄여줌
    return result

def preprocessing(df):
    # df['sentence_1'] = df['sentence_1'].apply(spacing)
    # df['sentence_1'] = df['sentence_1'].apply(spell_check)
    df['sentence_1'] = df['sentence_1'].apply(remove_repeat_text)

    # df['sentence_2'] = df['sentence_2'].apply(spacing)
    # df['sentence_2'] = df['sentence_2'].apply(spell_check)
    df['sentence_2'] = df['sentence_2'].apply(remove_repeat_text)
    return df
