"""Домашнее задание 1 по анализу данных."""

# Библиотека для работы с системой (чтением файлов и т.д.)
import os
# Чтение .env
from dotenv import load_dotenv
from langchain_gigachat import GigaChat

# from langchain.schema import HumanMessage
from langchain_core.messages import HumanMessage

# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)

from chromadb.config import Settings
# Платно(
# from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


load_dotenv()
token = os.getenv('GIGACHAT_TOKEN')

# Чтение токена, отключение проверки SSL-сертификатов и определение гигачата
llm = GigaChat(
    credentials=token,
    verify_ssl_certs=False,
    model="GigaChat:latest"
    )

# Простой ответ через Гигачат
question = ("Ответь кратко."
            "Как звали друга Геральта из книг Сапковского о ведьмаке?"
            "Подсказка: он бард и поэт."
            )
# llm([HumanMessage(content=question)]).content[0:200]
response = llm.invoke([HumanMessage(content=question)])
print(f"\nОтвет на вопрос 1: {response.content[0:200]}.\n")


# Загрузка файла
loader = TextLoader("Master_i_Margarita.txt", encoding="utf-8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # максимальный размер каждого чанка в символах
    # гарантия сохранения данных на перекрытиях
    chunk_overlap=200,  # перекрытие между чанками (в символах)
)
# RecursiveCharacterTextSplitter - умный разделитель текста
# 1. по двойным переводам строк (\n\n)
# 2. по одиночным переводам строк (\n)
# 3. по точкам (.)
# 4. по другим разделителям
documents = text_splitter.split_documents(documents)
print(f"Total documents: {len(documents)}")


# embeddings = GigaChatEmbeddings(
#     credentials=token, verify_ssl_certs=False
# )
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
db = Chroma.from_documents(
    documents,
    embeddings,
    client_settings=Settings(anonymized_telemetry=False),
)
