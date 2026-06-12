# Building a Vector store 
import os
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

with open('notes.txt','r') as f:
    text = f.read()

docs = [Document(page_content=text)]

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

chunks = splitter.split_text(text)

splitter2 = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

final_chunks = splitter2.split_documents(chunks)

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

vector_store = Chroma.from_documents(
    documents = final_chunks,
    embedding = embedding_model,
    persist_directory="vectorstore"
)

print(vector_store._collection.count())
