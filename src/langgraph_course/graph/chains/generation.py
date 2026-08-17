from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
"""Prompt standard per applicazioni RAG, con istruzioni per il modello e un esempio di domanda e risposta.
DEPRECTED"""
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\nContext:\n{context}"),
    ("human", "{question}"),
])
generation_chain = prompt | llm | StrOutputParser()