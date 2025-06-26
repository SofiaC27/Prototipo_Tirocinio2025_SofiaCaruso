import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase

# Recupera la chiave API e l'endpoint Groq
api_key = st.secrets["general"]["GROQ_LLM_KEY"]

llm = ChatOpenAI(
    model="llama3-8b-8192",
    temperature=0,
    openai_api_key=api_key,
    openai_api_base="https://api.groq.com/openai/v1",
)

# Inizializza connessione SQLite
db = SQLDatabase.from_uri("sqlite:///documents.db")
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)


def run_nl_query(question):
    """
    Funzione che esegue una query in linguaggio naturale sul database SQLite
    """
    return db_chain.invoke(question)
