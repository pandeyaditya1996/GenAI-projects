from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import pandas as pd
import os


df = pd.read_csv("realistic_restaurant_reviews.csv")

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"

add_documents = not os.path.exists(db_location)

if add_documents:

    documents = []
    ids = []

    for index, row in df.iterrows():
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata = {"rating": row["Rating"], "date": row["Date"]},
            id =str(index)
        )
        documents.append(document)
        ids.append(str(index))

vectorstore = Chroma(
    collection_name = "restaurant_reviews",
    embedding_function = embeddings,
    persist_directory = db_location
)

if add_documents:
    vectorstore.add_documents(documents, ids = ids)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})


