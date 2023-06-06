import io
import warnings

import docx
import streamlit as st

import model
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

if click:
    ready_text = model.process_text(resumeText)
    st.write(ready_text)

