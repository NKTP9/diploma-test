import csv
import io
import pickle
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
import streamlit as st
import docx
import joblib
import numpy as np
from catboost import CatBoostClassifier
import fasttext
import fasttext.util
import sklearn


def nltk2wn_tag(nltk_tag):
    """
    Эта функция используется для преобразования тегов частей речи,
    используемых в библиотеке NLTK (такие как 'JJ', 'VB', 'NN' и 'RB'),
    в теги частей речи WordNet (ADJ, VERB, NOUN и ADV соответственно),
    которые используются в библиотеке NLTK WordNet.
    Это необходимо для правильного лемматизации слов, поскольку WordNet использует теги частей речи для определения базовых
    форм слов (лемм)."""
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


def docx_to_text(file_path):
    document = docx.Document(file_path)
    text = ""
    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"
    return text


def get_highest_probability(prediction_proba):
    class_probabilities = prediction_proba[0]
    class_probabilities_pct = class_probabilities * 100

    highest_probability_index = np.argmax(class_probabilities_pct)
    highest_probability_pct = class_probabilities_pct[highest_probability_index]

    return "{:.2f}%".format(highest_probability_pct)


def csv_to_dict(csv_file_path):
    with open(csv_file_path, 'r') as file:
        reader = csv.DictReader(file)
        dictionary = {}
        for row in reader:
            for key, value in row.items():
                if key in dictionary:
                    dictionary[key].append(value)
                else:
                    dictionary[key] = [value]
    return dictionary


# Скачиваем Русскую модель
fasttext.util.download_model('ru', if_exists='ignore')

# Загружаем готовые FastText эмбединги
fasttext_model = fasttext.load_model('cc.ru.300.bin')


def fasttext_vectorizer(texts):
    """
    Данная функция векторизует текстовые данные с помощью предобученной модели FastText,
    которая преобразует тексты в векторы фиксированной размерности.
    Входным параметром является список текстов, на выходе функция возвращает массив с векторными представлениями текстов,
    где каждая строка соответствует одному тексту. Каждый вектор имеет размерность,
    равную размерности пространства эмбеддингов, которое определяется моделью FastText."""
    embeddings = np.zeros((len(texts), fasttext_model.get_dimension()))
    for i, text in enumerate(texts):
        embeddings[i] = fasttext_model.get_sentence_vector(text)
    return embeddings


def get_highest_probability(prediction_proba):
    class_probabilities = prediction_proba[0]
    class_probabilities_pct = class_probabilities * 100

    highest_probability_index = np.argmax(class_probabilities_pct)
    highest_probability_pct = class_probabilities_pct[highest_probability_index]

    return "{:.2f}%".format(highest_probability_pct)


st.title("Candidate Selection Tool")

st.subheader("NLP Based Resume Screening")

st.caption(
    "Aim of this project is to check whether a candidate is qualified for a role based his or her education, experience, and other information captured on their resume. In a nutshell, it's a form of pattern matching between a job's requirements and the qualifications of a candidate based on their resume.")

uploadedResume = st.file_uploader("Upload resume", type="docx")

click = st.button("Process")

try:
    if uploadedResume is not None:
        resumeBytes = uploadedResume.read()
        resumeDoc = docx.Document(io.BytesIO(resumeBytes))
        resumeText = '\n'.join([para.text for para in resumeDoc.paragraphs])
        st.write(resumeText)
except Exception as e:
    st.warning("Error: {}".format(e))

if click:
    with open('pipeline.pkl', 'rb') as f:
        pipeline = pickle.load(f)

    ready_text = preprocess_text(resumeText)
    prediction = pipeline.predict([ready_text])

    job_dict = csv_to_dict('dictionary.csv')

    classifier = CatBoostClassifier(iterations=100, eval_metric='Accuracy', use_best_model=True, random_seed=19)
    vectorized_text = fasttext_vectorizer([ready_text])
    prediction_proba = pipeline.predict_proba(vectorized_text)
    print(prediction_proba)

    label_prob = get_highest_probability(prediction_proba)
    label_num = prediction.item()

    if label_num in job_dict:
        print("Вы подходите на вакансию \"" + prediction + "\". Соответствие вашего резюме описанию: " + label_prob)
    else:
        print('Error: label not found in dictionary.')
