import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from Modules.ocr_groq import load_prompt


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
        use_query_checker=True,
    )

    return db_chain


def is_question_valid_for_db(question, db_schema, llm):
    """
    Funzione per verificare se una domanda in linguaggio naturale è pertinente rispetto allo schema di un database SQL
    - Carica un prompt da file esterno, dove lo schema e la domanda vengono inseriti dinamicamente
    - Il prompt viene passato all’LLM, che deve rispondere solo con "true" o "false"
    - Costruisce una catena con il prompt, il modello LLM e il parser
    - Esegue la catena passando schema e domanda come input
    - La risposta viene convertita in un booleano Python per determinare se la domanda è valida
    :param question: stringa contenente la domanda in linguaggio naturale da verificare
    :param db_schema: stringa rappresentante lo schema SQL del database da consultare
    :param llm: modello LLM compatibile con LangChain
    :return: True se la domanda è pertinente allo schema, altrimenti False
    """
    prompt_text = load_prompt("Modules/AI_prompts/validity_prompt.txt")

    prompt = PromptTemplate.from_template(prompt_text)

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "question": question,
        "schema": db_schema
    })

    print("validità di:", question, "=>", result)

    return result.strip().lower() == "true"


def run_nl_query(question, chain):
    """
    Funzione per elaborare una domanda in linguaggio naturale ed eseguire una query SQL attraverso una catena LangChain
    - Recupera lo schema del database
    - Recupera il modello LLM utilizzato dalla SQLDatabaseChain
    - Verifica se la domanda è compatibile con lo schema del database tramite una validazione con l'LLM
    - Se non è compatibile, restituisce un dizionario con un messaggio di risposta e tutti gli altri campi nulli
    - Se è compatibile: esegue la catena LLM sul prompt della query, estrae i passaggi intermedi dalla
       risposta ottenuta (la query SQL generata, il risultato SQL del database e la risposta finale del modello),
       costruisce e restituisce un dizionario strutturato con tutti i dati
    :param question: stringa contenente la domanda in linguaggio naturale dell'utente
    :param chain: istanza di SQLDatabaseChain configurata con un LLM e un database SQL
    :return: dizionario con la domanda dell'utente, i passaggi intermedi estratti e lo stato della domanda
    """
    db_schema = chain.database.get_table_info()
    llm = chain.llm_chain.llm

    # Validazione della domanda
    if not is_question_valid_for_db(question, db_schema, llm):
        return {
            "question": question,
            "sql_query": None,
            "sql_result": None,
            "answer": None,
            "status": "invalid_question"
        }

    # Esecuzione della catena
    response = chain.invoke({"query": question})

    intermediate = response.get("intermediate_steps", [])

    query_sql = intermediate[2]["sql_cmd"]
    query_result = intermediate[3]
    final_answer = intermediate[5]

    output = {
        "question": question,
        "sql_query": query_sql,
        "sql_result": query_result,
        "answer": final_answer,
        "status": "valid_question"
    }

    return output


def render_llm_interface():
    """
    Funzione per visualizzare l'interfaccia di interrogazione su database SQL tramite LLM
    - Inizializza la catena SQLDatabaseChain tramite chiave API
    - Mostra un messaggio di info con la descrizione del database per aiutare l'utente a fare domande pertinenti
    - Mostra una casella di input testuale per inserire una domanda in linguaggio naturale
    - Esegue una query SQL tramite il modello LLM se la domanda è valida
    - Visualizza i risultati strutturati: domanda, query SQL generata, risultato grezzo del database e risposta testuale
    - Se la domanda è incompatibile con il database oppure se la domanda è valida, ma non ci sono risultati
      mostra dei warning
    """
    llm_key = st.secrets["general"]["GROQ_LLM_KEY"]

    if "llm_chain" not in st.session_state:
        st.session_state.llm_chain = init_db_chain(llm_key)

    if "llm_result" not in st.session_state:
        st.session_state.llm_result = None

    st.info("The database stores information extracted from receipts: it includes data about"
            " uploaded receipt images, store and transaction details, and the list of purchased items"
            " with potential discounts. It is designed to answer questions about purchases, stores, "
            " prices, dates, products, and payment methods.")

    question = st.text_input("Enter a question:", key="nl_input")

    if question:
        res = run_nl_query(question, st.session_state.llm_chain)
        st.session_state.llm_result = res

    if "llm_result" in st.session_state and st.session_state.llm_result:
        res = st.session_state.llm_result

        st.markdown("# Natural language question:")
        st.write(res["question"])

        if res["status"] == "invalid_question":
            st.warning("The question is not compatible with the information in the database. Please"
                       " try asking a different, more suitable question")
        elif not res["sql_result"] and res["sql_query"]:  # Query valida, ma risultato vuoto
            st.warning("The question is compatible with the database, but no matching data was "
                       " found. Try changing the filters")
        else:
            st.markdown("# Generated SQL query:")
            st.code(res["sql_query"], language="sql")

            st.markdown("# Raw database result:")
            st.write(res["sql_result"])

            st.markdown("# Model-generated answer:")
            st.text(res["answer"])
