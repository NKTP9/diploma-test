import model
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
import numpy as np
import fasttext.util
import warnings
import pandas as pd

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


def fasttext_vectorizer(texts, fasttext_model):
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


def group_jobs_by_label(df):
    label_groups = {}
    for label in df['target'].unique():
        label_df = df[df['target'] == label]
        label_df['name'] = label_df['name'].apply(lambda x: re.sub(r'\(.*\)', '', x.strip()).capitalize())
        common_job = get_common_job(label_df['name'])
        label_groups[label] = common_job
    return label_groups


def get_common_job(professions):
    common_job = None
    for profession in professions:
        if common_job is None:
            common_job = profession
        elif common_job not in profession:
            common_job = None
            break
    if common_job is None:
        common_job = professions.iloc[0]
    return common_job.capitalize()

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Скрининг резюме")

st.title("Система автоматического скрининга резюме")

st.caption("""Наша система позволяет сканировать текстовый файл формата DOCX и подготавливать его для классификации 
моделью. После анализа резюме система определяет, какой вакансии соответствует данный документ и насколько точно он 
соответствует требованиям, описанным в вакансии.
Для того, чтобы воспользоваться системой, необходимо загрузить файл формата DOCX. После загрузки 
файла, система считывает его содержимое и обрабатывает текст. Классификатор определяет, 
какая работа лучше всего подходит для данного резюме и насколько оно соответствует описанию каждой вакансии. В 
случае, если система определяет, что резюме подходит для конкретной вакансии, она выводит соответствующую информацию. 
Наша система может быть полезна для тех, кто ищет работу и хочет определить, на какие вакансии стоит 
подавать резюме, а также для работодателей, которые хотят автоматизировать процесс отбора резюме.""")

uploadedResume = st.file_uploader("Загрузите резюме в формате DOCX", type="docx")

click = st.button("Process")

try:
    if uploadedResume is not None:
        resumeBytes = uploadedResume.read()
        resumeDoc = docx.Document(io.BytesIO(resumeBytes))
        resumeText = '\n'.join([para.text for para in resumeDoc.paragraphs])
except Exception as e:
    st.warning("Error: {}".format(e))

fasttext_model = fasttext.load_model('cc.ru.300.bin')

csv_file = 'train.csv'
df = pd.read_csv(csv_file)

job_dict = model.group_jobs_by_label(df)

if click:
    with open('catboost_classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)

    resumeText = model.docx_to_text('Resume1.docx')

    ready_text = model.preprocess_text(resumeText)
    vectorized_text = model.fasttext_vectorizer([ready_text], fasttext_model)
    prediction = classifier.predict(vectorized_text)
    prediction_proba = classifier.predict_proba(vectorized_text)

    label_prob = model.get_highest_probability(prediction_proba)
    label_num = prediction.item()

    if label_num in job_dict:
        st.write("Ваша подходящая вакансия - " + job_dict[label_num]
                 + ". Соответствие вашего резюме описанию: " + label_prob)
    else:
        st.write('Ошибка, резюме не было обработано корректно. Скорее всего, оно не подходит под нашу систему...')
