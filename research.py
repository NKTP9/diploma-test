#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Импортируем библотеки
import pandas as pd
import numpy as np


# In[20]:


# Считываем данные
data = pd.read_csv('train.csv')


# In[21]:


# Посмотрим на данные
data


# In[22]:


# Посмотрим количество текстов пренадлежащих разным классам
data['target'].value_counts()


# ## Построим первичную модель

# In[5]:


# Импортируем нужные билиотеки
import pandas as pd
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from pymystem3 import Mystem


# In[23]:


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

    # лемматизация
    m = Mystem()
    lemmatized_words = [''.join(m.lemmatize(w)).strip() for w in words]

    return ' '.join(lemmatized_words)


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Загружаем данные\ndata = pd.read_csv('train.csv', index_col='index')\n\n# комбинируем название и описание в один текстовый столбец\ndata['text'] = data['name'].fillna('') + ' ' + data['description'].fillna('')\ndata.drop(['name', 'description'], axis=1, inplace=True)\n\n# Обработка текста\ndata['text'] = data['text'].apply(preprocess_text)")


# In[9]:


data.head()


# In[10]:


# Делим данные на тренировочный и тестовый датасет
X_train, X_val, y_train, y_val = train_test_split(data['text'], data['target'], test_size=0.2, random_state=42)


# In[ ]:


# Делим данные на тренировочный и тестовый датасет
X_train, X_val, y_train, y_val = train_test_split(data['text'], data['target'], test_size=0.2, random_state=42)

# Создаем цепочку из векторизатора и модели логистической регрессии
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression(solver='liblinear')),
])

# Тренируем модель
pipeline.fit(X_train, y_train)

# Оцениваем модель на валидационом датасете
y_pred = pipeline.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_val, y_pred))


# ## Используем предобученные векторизаторы

# In[10]:


# Устанавливаем библиотеку
if 0:
    get_ipython().system('pip install fasttext')


# In[1]:


# Импортируем библиотеки
import fasttext
import fasttext.util

# Скачиваем русскую модель
fasttext.util.download_model('ru', if_exists='ignore')


# In[2]:


# Загружаем готовые FastText числовые векторы
fasttext_model = fasttext.load_model('cc.ru.300.bin')

def fasttext_vectorizer(texts):
    '''
    Данная функция векторизует текстовые данные с помощью предобученной модели FastText, 
    которая преобразует тексты в векторы фиксированной размерности. 
    Входным параметром является список текстов, на выходе функция возвращает массив с векторными представлениями текстов, 
    где каждая строка соответствует одному тексту. Каждый вектор имеет размерность, 
    равную размерности пространства эмбеддингов, которое определяется моделью FastText.'''
    embeddings = np.zeros((len(texts), fasttext_model.get_dimension()))
    for i, text in enumerate(texts):
        embeddings[i] = fasttext_model.get_sentence_vector(text)
    return embeddings


# In[7]:


vect_text = fasttext_vectorizer('работа рядом домом учебой график стабильная выплата заработной платы раза месяц')
vect_text


# In[13]:


# Применение функций векторизаторов к данным
X_train_embeddings = fasttext_vectorizer(X_train)
X_val_embeddings = fasttext_vectorizer(X_val)

# Тренируем модель Логистической Регрессии
classifier = LogisticRegression(solver='liblinear')
classifier.fit(X_train_embeddings, y_train)

# Оцениваем модель на валидационом датасете
y_pred = classifier.predict(X_val_embeddings)
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_val, y_pred))


# In[12]:


# Создаем копию датасета 
data_new = data.copy()


# In[13]:


# Создадим датасет, где у нас у класса минимум 3 элемента
col = data_new['target'].value_counts()
bad_classes = col.loc[col < 3].index


# In[14]:


# Присвоим -2
data_new.loc[data_new['target'].isin(bad_classes), 'target'] = -2


# In[15]:


# Смотрим на результат
data_new['target'].value_counts()


# In[16]:


# Уберем класс -1 и -2
data_new = data_new.loc[(data_new['target'] != -1) & (data_new['target'] != -2)]


# In[ ]:


from catboost import CatBoostClassifier
from sklearn.utils import resample
import pickle

# Делим данные на тренировочную и тестовую выборку
X_train, X_val, y_train, y_val = train_test_split(data_new['text'], data_new['target'], test_size=0.2, 
                                                  random_state=19, stratify=data_new['target'])

# Конвертируем текста в эмбединги
X_train_embeddings = fasttext_vectorizer(X_train)
X_val_embeddings = fasttext_vectorizer(X_val)

# Тренируем CatBoost classifier
classifier = CatBoostClassifier(iterations=100, eval_metric='Accuracy', use_best_model=True, random_seed=19)
classifier.fit(X_train_embeddings, y_train, eval_set=(X_val_embeddings, y_val))

# Сохраняем обученную модель с использованием pickle
pickle_file = "catboost_classifier.pkl"
with open(pickle_file, "wb") as f:
    pickle.dump(classifier, f)

# Оценим модель
y_pred = classifier.predict(X_val_embeddings)
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_val, y_pred))

