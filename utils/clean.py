import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

def remove_punctuation_and_special_chars(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def remove_stopwords(text, language='english'):
    stop_words = set(stopwords.words(language))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

def clean_text(text, remove_stop=True, language='english'):
    text = text.lower()
    text = remove_punctuation_and_special_chars(text)
    
    if remove_stop:
        text = remove_stopwords(text, language)
    
    return text

def clean_texts(texts, remove_stop=True, language='english'):
    return [clean_text(text, remove_stop, language) for text in texts]