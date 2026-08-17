from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

urls = [
  "https://lilianweng.github.io/posts/2023-06-23-agent/",
  "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
  "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/"
]

"""Con un limite impostato così alto il  taglio è fittizzio, ma serve per avere un unico documento per ogni url, così da poterlo poi tagliare in chunk più piccoli"""
docs = [WebBaseLoader(url).load() for url in urls]

"""docs è una lista di liste, quindi la flatteno in una lista unica"""
docs_list = [doc for sublist in docs for doc in sublist]  # Flatten the list of lists

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=250, chunk_overlap=0)

docs_splits = text_splitter.split_documents(docs_list)

# vector_store = Chroma.from_documents(documents=docs_splits, collection_name="rag-chroma", embedding=OpenAIEmbeddings(), persist_directory="./.chroma")

"""Serve per recuperare i documenti dalla collection di Chroma, in modo da poter fare il retrieval e poi la generazione della risposta"""
retriver = Chroma(collection_name="rag-chroma", embedding_function=OpenAIEmbeddings(), persist_directory="./.chroma").as_retriever()