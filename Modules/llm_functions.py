import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_sql_query_chain


from Modules.ocr_groq import load_prompt


def init_chain(api_key):
    """
    Funzione per inizializzare la catena LangChain per interrogazioni in linguaggio naturale su database SQL
    - Configura il modello LLM llama3 tramite endpoint Groq, utilizzando l'API key fornita
    - Crea la connessione al database locale SQLite
    - Costruisce una catena LangChain che, dato un input testuale in linguaggio naturale, restituisce
      una query SQL come stringa, senza eseguirla
    :param api_key: chiave API per autenticare le richieste al provider Groq (OpenAI compatibile)
    :return: una catena Runnable che accetta una domanda e restituisce una query SQL
    :return: oggetto SQLDatabase connesso al database locale
    :return: modello LLM configurato per la generazione delle query
    """
    llm = ChatOpenAI(
        model="llama3-8b-8192",
        temperature=0,
        openai_api_key=api_key,
        openai_api_base="https://api.groq.com/openai/v1",
    )

    db = SQLDatabase.from_uri("sqlite:///documents.db")

    sql_only_prompt = PromptTemplate.from_template("""
    Hai accesso a un database SQL con la seguente struttura:

    {table_info}

    Domanda: {input}

    Scrivi una query SQL compatibile con il database per rispondere alla domanda.

    Se la domanda non è compatibile con le informazioni disponibili nel database non generare 
    alcuna query e lascia la risposta vuota.

    Restituisci solo la query SQL, senza spiegazioni o testo aggiuntivo.

    Se possibile, limita il numero di risultati a massimo {top_k} righe.
    """)

    query_chain = create_sql_query_chain(
        llm=llm,
        db=db,
        prompt=sql_only_prompt,
        k=100
    )

    return query_chain, db, llm


def is_query_valid_for_db(sql_query, db_schema, llm):
    """
    Funzione per verificare se una query SQL generata è compatibile con lo schema di un database SQL
    - Carica un prompt da file esterno, dove lo schema e la query vengono inseriti dinamicamente
    - Il prompt viene passato all’LLM, che deve rispondere solo con "true" o "false"
    - Costruisce una catena composta dal prompt, dal modello LLM e da un parser per l’output testuale
    - Esegue la catena passando lo schema e la query come input
    - La risposta viene convertita in un booleano Python per determinare se la query è semanticamente valida
    :param sql_query: stringa contenente la query SQL generata da validare
    :param db_schema: stringa rappresentante lo schema SQL del database da consultare
    :param llm: modello LLM compatibile con LangChain
    :return: True se la query è semanticamente compatibile con lo schema, altrimenti False
    """
    prompt_text = load_prompt("Modules/AI_prompts/validity_prompt.txt")

    prompt = PromptTemplate.from_template(prompt_text)

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "sql_query": sql_query,
        "schema": db_schema
    })

    print("Query validity:", result)

    return result.strip().lower() == "true"


def format_model_answer(sql_query, raw_result, llm):
    """
    Funzione per generare una risposta formattata e leggibile in italiano a partire da una query SQL e dal suo risultato
    - Carica da file un prompt che include le istruzioni per la formattazione e la traduzione in italiano
    - Inserisce dinamicamente la query SQL e il risultato grezzo all'interno del prompt
    - Invia il prompt al modello LLM per ottenere una risposta testuale coerente, chiara e adatta all’utente
    :param sql_query: stringa contenente la query SQL generata
    :param raw_result: risultato grezzo ottenuto dall’esecuzione della query sul database
    :param llm: istanza del modello LLM da utilizzare per la riformattazione
    :return: stringa con la risposta finale formattata in italiano
    """
    prompt_text = load_prompt("Modules/AI_prompts/format_answer_prompt.txt")

    full_input = prompt_text.format(
        query=sql_query,
        result=raw_result
    )

    result = llm.invoke(full_input)
    return result.content.strip()


def run_nl_query(question, query_chain, db, llm):
    """
     Funzione per elaborare una domanda in linguaggio naturale ed eseguire una query SQL tramite LangChain
    - Recupera lo schema SQL del database corrente
    - Genera una query SQL a partire dalla domanda usando la catena creata
    - Verifica se la query generata è semanticamente compatibile con lo schema del database tramite un LLM esterno
    - Se la query è incompatibile, restituisce uno stato "invalid_question" e non viene eseguita
    - Se la query è valida: viene eseguita sul database, e il risultato grezzo viene passato al modello per generare
      una risposta formattata in italiano
    - In caso di errore durante il processo, restituisce uno stato "error" con il messaggio dell'eccezione

    :param question: stringa contenente la domanda in linguaggio naturale dell'utente
    :param query_chain: catena LangChain che genera una query SQL a partire da una domanda
    :param db: istanza SQLDatabase connessa al database da interrogare
    :param llm: istanza LLM compatibile con LangChain, usata per validazione e formattazione
    :return: dizionario con la domanda, la query SQL generata, il risultato grezzo, la risposta formattata e lo stato
    """
    try:
        db_schema = db.get_table_info()
        sql_query = query_chain.invoke({"question": question})

        # Validazione post-query
        if not is_query_valid_for_db(sql_query, db_schema, llm):
            return {
                "question": question,
                "sql_query": None,
                "raw_result": None,
                "answer": None,
                "status": "invalid_question"
            }
        else:
            # Esegui la query
            raw_result = db.run(sql_query)

            # Genera la risposta formattata
            formatted_answer = format_model_answer(sql_query, raw_result, llm)

            return {
                "question": question,
                "sql_query": sql_query,
                "raw_result": raw_result,
                "answer": formatted_answer,
                "status": "valid_question"
            }

    except Exception as e:
        return {
            "question": question,
            "sql_query": None,
            "raw_result": None,
            "answer": None,
            "status": "error",
            "error_message": str(e)
        }


def render_llm_interface():
    """
    Funzione per visualizzare l'interfaccia per interrogazioni in linguaggio naturale su database SQL tramite LLM
    - Recupera la chiave API e inizializza la catena LangChain per la generazione di query SQL e il modello LLM
    - Mostra un messaggio informativo con la descrizione del database per guidare l’utente nella formulazione
      delle domande
    - Visualizza una selectbox con esempi di domande, che funge anche da campo di input testuale
    - Esegue la funzione per generare la query SQL, validarla, eseguirla e formattare la risposta
    - Visualizza i risultati strutturati: stato della domanda, domanda originale, query SQL generata, risultato grezzo
      e risposta finale
    - In caso di domanda non compatibile con lo schema del database, mostra un messaggio di avviso
    - In caso di errore durante l’elaborazione, mostra il messaggio dell’eccezione sollevata
    """
    llm_key = st.secrets["general"]["GROQ_LLM_KEY"]

    if "query_chain" not in st.session_state or "llm" not in st.session_state or "db" not in st.session_state:
        query_chain, db, llm = init_chain(llm_key)
        st.session_state.query_chain = query_chain
        st.session_state.llm = llm
        st.session_state.db = db

    if "llm_result" not in st.session_state:
        st.session_state.llm_result = None
    if "last_rendered_answer" not in st.session_state:
        st.session_state.last_rendered_answer = None
    if "submitted_question" not in st.session_state:
        st.session_state.submitted_question = None

    st.info("The database stores information extracted from receipts: it includes data about"
            " uploaded receipt images, store and transaction details, and the list of purchased items"
            " with potential discounts. It is designed to answer questions about purchases, stores, "
            " prices, dates, products, and payment methods.")

    examples = [
        "Mostrami i primi 15 scontrini caricati nel 2025",
        "Mostrami i primi 10 acquisti effettuati nel 2025",
        "Elenca i prodotti per cui è stato applicato uno sconto",
        "Qual è la somma totale delle spese effettuate nel mese di marzo?",
        "Quali prodotti sono stati acquistati più di una volta in giorni diversi?",
        "In quale mese del 2025 ho speso di più in totale?",
        "Quali negozi ho visitato più spesso?",
        "Qual è stato il metodo di pagamento più usato nei miei acquisti?",
        "Mostrami tutti i prodotti acquistati in contanti",
        "Quali sono i prodotti più acquistati in termini di quantità totale?"
    ]

    user_question = st.selectbox(
        "Enter a question or choose one from the examples below:",
        options=examples,
        index=None,
        placeholder="Type or select a question...",
        accept_new_options=True,
        key="nl_input"
    )

    # Esegue la query solo se la domanda è nuova o diversa dalla precedente
    if user_question and user_question != st.session_state.submitted_question:
        st.session_state.submitted_question = user_question
        res = run_nl_query(user_question, st.session_state.query_chain, st.session_state.db, st.session_state.llm)
        st.session_state.llm_result = res
        st.session_state.last_rendered_answer = res["answer"]

    if "llm_result" in st.session_state and st.session_state.llm_result:
        res = st.session_state.llm_result

        st.markdown("# Query status:")
        st.write(res['status'])

        match res["status"]:
            case "valid_question":
                st.markdown("# Natural language question:")
                st.write(res["question"])

                st.markdown("# Generated SQL query:")
                st.code(res["sql_query"], language="sql")

                st.markdown("# Raw query result:")
                st.write(res["raw_result"])

                st.markdown("# Model-generated answer:")
                st.text(res["answer"])

            case "invalid_question":
                st.warning("The question is not compatible with the information in the database. Please"
                           " try asking a different, more suitable question")

            case "error":
                error_msg = res.get("error_message", "An unexpected error occurred")
                st.error(f"An error occurred while answering the question using SQL:\n\n{error_msg}")
