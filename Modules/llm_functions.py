import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase


def init_db_chain(api_key):
    """
    Funzione per inizializzare la catena LangChain per interrogazioni in linguaggio naturale su database SQL
    - Configura il modello LLM llama3 tramite endpoint Groq, utilizzando l'API key fornita
    - Crea la connessione al database locale SQLite
    - Genera una SQLDatabaseChain che permette al LLM di: interpretare domande in linguaggio naturale, tradurle
      in query SQL valide, eseguire le query sul database
    - Abilita l'output dettagliato (verbose) e il recupero dei passaggi intermedi per finalità di trasparenza
    - Attiva il query checker per evitare errori o query ambigue
    :param api_key: chiave API per autenticare le richieste al provider Groq (OpenAI compatibile)
    :return: oggetto SQLDatabaseChain configurato per gestire query NL → SQL → risultati
    """
    llm = ChatOpenAI(
        model="llama3-8b-8192",
        temperature=0,
        openai_api_key=api_key,
        openai_api_base="https://api.groq.com/openai/v1",
    )

    db = SQLDatabase.from_uri("sqlite:///documents.db")

    db_chain = SQLDatabaseChain.from_llm(
        llm=llm,
        db=db,
        verbose=True,
        return_intermediate_steps=True,
        use_query_checker=True
    )

    return db_chain


def run_nl_query(question, chain):
    """
    Funzione per elaborare una domanda in linguaggio naturale ed eseguire una query SQL attraverso LangChain
    - Utilizza la catena `SQLDatabaseChain` per interpretare la domanda dell'utente
    - Il modello LLM genera una query SQL a partire dal linguaggio naturale
    - Viene eseguita la query sul database configurato e ne viene restituito il risultato
    - Recupera anche i passaggi intermedi per estrarre la query SQL generata
    :param question: stringa contenente la domanda in linguaggio naturale dell'utente
    :param chain: oggetto SQLDatabaseChain inizializzato con LLM e database SQLite
    :return: dizionario contenente la query SQL generata e utilizzata e il risultato finale prodotto dal
             modello LLM dopo l'esecuzione della query
    """
    response = chain(question)

    output = {
        "query": response.get("intermediate_steps", [""])[-1],  # Ultima query SQL generata
        "result": response.get("result")  # Risposta finale dell’LLM
    }

    return output
