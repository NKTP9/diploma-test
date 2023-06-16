import csv
import pickle
import re
import warnings

import docx
import fasttext.util
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

warnings.filterwarnings("ignore")

# Предзагрузка векторизатора FastText
fasttext_model = fasttext.load_model('cc.ru.300.bin')


# Метод определения части речи слова
def nltk2wn_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


# Метод предобработки текста: токенизация, нормализация, лемматизация, удаление стоп-слов
def preprocess_text(text):
    # удаление HTML/XML тэгов
    text = re.sub('<[^>]+>', '', text)

    # удаление символов кроме букв
    text = re.sub('[^a-zA-Zа-яА-Я]', ' ', text)

    # приведение текста к нижнему регистру
    text = text.lower()

    # токенизация
    words = word_tokenize(text)

    # Удаление стоп слов
    stop_words = set(stopwords.words('russian'))
    words = [w for w in words if w not in stop_words]

    # стемминг и лемматизация
    stemmer = SnowballStemmer('russian')
    lemmatizer = WordNetLemmatizer()
    stemmed_words = [stemmer.stem(w) for w in words]
    tagged_words = nltk.pos_tag(words)
    lemmatized_words = [lemmatizer.lemmatize(w, pos=nltk2wn_tag(t)) if nltk2wn_tag(t) is not None else w for w, t in
                        tagged_words]

    return ' '.join(lemmatized_words)


# Метод преобразования DOCX-файла в текст
def docx_to_text(file_path):
    document = docx.Document(file_path)
    text = ""
    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"
    return text


# Метод получения самого высокого процента вакансии
def get_highest_probability(prediction_proba):
    class_probabilities = prediction_proba[0]
    class_probabilities_pct = class_probabilities * 100

    highest_probability_index = np.argmax(class_probabilities_pct)
    highest_probability_pct = class_probabilities_pct[highest_probability_index]

    return "{:.2f}%".format(highest_probability_pct)


# Метод преобразования csv-файла с профессиями в словарь
def csv_to_dict(file_name):
    with open(file_name, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        dictionary = {int(rows[0]): rows[1] for rows in reader}
    return dictionary


# Метод преобразования текста в числовые векторы
def fasttext_vectorizer(texts, fasttext_model):
    embeddings = np.zeros((len(texts), fasttext_model.get_dimension()))
    for i, text in enumerate(texts):
        embeddings[i] = fasttext_model.get_sentence_vector(text)
    return embeddings


# Основной метод модели машинного обучения
def process_text(text):
    with open('catboost_classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)

    job_dict = csv_to_dict('dictionary.csv')

    ready_text = preprocess_text(text)
    vectorized_text = fasttext_vectorizer([ready_text], fasttext_model)
    prediction = classifier.predict(vectorized_text)
    prediction_proba = classifier.predict_proba(vectorized_text)

    label_prob = get_highest_probability(prediction_proba)
    label_num = prediction.item()

    if label_num in job_dict:
        return r'Ваше присланное резюме было обработано. Исходя из результатов предсказания, вы больше всего ' \
               r'подходите на профессию "' + job_dict[label_num] + '" с точностью ' + label_prob + '.'
    else:
        return r'Ваше резюме не было обработано, попробуйте снова'
