import streamlit as st
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_sql_query_chain
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType


from Modules.ocr_groq import load_prompt


def init_chain(llm, db):
    """
    Funzione per inizializzare la catena LangChain per interrogazioni in linguaggio naturale su database SQL
    - Carica un prompt da file per la generazione della query
    - Costruisce una catena LangChain che restituisce una query SQL come stringa.
    :param llm: modello LLM
    :param db: oggetto SQLDatabase connesso al database locale
    :return: una catena Runnable
    """
    prompt_text = load_prompt("Modules/AI_prompts/sql_generation_prompt.txt")
    sql_only_prompt = PromptTemplate.from_template(prompt_text)
    query_chain = create_sql_query_chain(
        llm=llm,
        db=db,
        prompt=sql_only_prompt,
        k=100
    )

    return query_chain


def is_question_valid_for_db(question, llm, db_schema):
    """
    Funzione per verificare se una domanda in linguaggio naturale è semanticamente compatibile con
    lo schema del database
    - Carica un prompt da file esterno
    - Costruisce una catena LangChain con il prompt, il modello LLM e un parser
    - Passa la domanda e lo schema al modello
    - Interpreta la risposta come booleano
    :param question: domanda in linguaggio natuarale dell'utente
    :param llm: modello LLM
    :param db_schema: schema del database locale
    :return: True se la domanda è compatibile, altrimenti False
    """
    prompt_text = load_prompt("Modules/AI_prompts/question_validity_prompt.txt")

    prompt = PromptTemplate.from_template(prompt_text)
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "question": question,
        "schema": db_schema
    })

    return "true" in result.strip().lower()


def is_query_valid_for_db(sql_query, llm, db_schema):
    """
    Funzione per verificare se una query SQL generata è compatibile con lo schema del database
    - Carica un prompt da file esterno
    - Costruisce una catena LangChain con il prompt, il modello LLM e un parser.
    - Passa la query e lo schema al modello
    - Interpreta la risposta come booleano
    :param sql_query: query SQL generata da validare
    :param llm: modello LLM
    :param db_schema: schema del database locale
    :return: True se la query è compatibile, altrimenti False
    """
    prompt_text = load_prompt("Modules/AI_prompts/query_validity_prompt.txt")

    prompt = PromptTemplate.from_template(prompt_text)

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "sql_query": sql_query,
        "schema": db_schema
    })

    return "true" in result.strip().lower()


def format_model_answer(raw_result, llm):
    """
    Funzione per generare una risposta formattata e tradotta in italiano a partire dal risultato di una query SQL
    - Controlla se il risultato della query è vuoto e in caso da un messaggio di nessun risultato
    - Carica un prompt da file esterno
    - Inserisce dinamicamente la query e il risultato nel prompt
    - Invia il prompt al modello LLM
    :param raw_result: risultato grezzo della query eseguita sul database
    :param llm: modello LLM
    :return: stringa con la risposta finale formattata in italiano
    """
    if raw_result == "[]":
        return ("La richiesta è stata compresa ed elaborata correttamente, ma la query non ha restituito"
                " alcun risultato. Non sono stati trovati dati corrispondenti ai criteri specificati."
                " Potresti provare a modificare i parametri della ricerca per ottenere risultati diversi.")

    prompt_text = load_prompt("Modules/AI_prompts/format_answer_prompt.txt")
    formatted_prompt = prompt_text.format(result=raw_result)
    response = llm.invoke(formatted_prompt)

    return response.content.strip()


def build_question_validator_tool(llm, db_schema):
    """
    Funzione che crea un tool LangChain che valida semanticamente una domanda rispetto allo schema
    del database
    - Recupera lo schema dal database
    - Usa is_question_valid_for_db per la valutazione
    :param llm: modello LLM
    :param db_schema: schema dell'oggetto SQLDatabase connesso al database locale
    :return: oggetto Tool utilizzabile da un agente per validare le domande
    """
    def validate_question(question):
        return is_question_valid_for_db(question, llm, db_schema)

    return Tool(
        name="QuestionValidator",
        func=validate_question,
        description="Valida se una domanda in linguaggio naturale è semanticamente compatibile con lo"
                    " schema del database"
    )


def build_sql_query_tool(llm, db):
    """
    Funzione che crea un tool LangChain che genera una query SQL da una domanda in linguaggio naturale
    - Inizializza tramite init_chain la catena che restituisce la query
    :param llm: modello LLM
    :param db: oggetto SQLDatabase connesso al database locale
    :return: oggetto Tool utilizzabile da un agente che restituisce una query SQL come stringa
    """
    query_chain = init_chain(llm, db)

    def generate_sql(question):
        return query_chain.invoke({"question": question})

    return Tool(
        name="SQLQueryGenerator",
        func=generate_sql,
        description="Genera una query SQL a partire da una domanda in linguaggio naturale. Non esegue"
                    " la query"
    )


def build_query_validator_tool(llm, db_schema):
    """
    Funzione che crea un tool LangChain che verifica la compatibilità semantica di una query SQL con
    lo schema del database
    - Usa is_query_valid_for_db per la valutazione
    :param llm: modello LLM
    :param db_schema: schema dell'oggetto SQLDatabase connesso al database locale
    :return: oggetto Tool utilizzabile da un agente per validare le query
    """
    def validate_query(sql_query):
        return is_query_valid_for_db(sql_query, llm, db_schema)

    return Tool(
        name="QueryValidator",
        func=validate_query,
        description="Valida se una query SQL è semanticamente compatibile con lo schema del database"
    )


def build_query_executor_tool(db):
    """
    Funzione che crea un tool LangChain che esegue una query SQL sul database locale
    - Usa db.run() per eseguire la query
    - Se la query non restituisce un risultato, ritorna "[]"
    :param db: oggetto SQLDatabase connesso al database locale
    :return: oggetto Tool utilizzabile da un agente che restituisce il risultato grezzo della query
    """
    def execute_query(sql_query):
        try:
            result = db.run(sql_query)
            return result if result else "[]"
        except Exception as e:
            return f"Error during query execution: {str(e)}"

    return Tool(
        name="QueryExecutor",
        func=execute_query,
        description="Esegue una query SQL sul database e restituisce il risultato grezzo"
    )


def build_answer_formatter_tool(llm):
    """
    Funzione che crea un tool LangChain che formatta e traduce in italiano la risposta del modello con il
    risultato di una query SQL
    - Usa format_model_answer per generare la risposta
    :param llm: modello LLM
    :return: oggetto Tool utilizzabile da un agente che restituisce la risposta come stringa formattata
    """
    def format_answer(raw_result):
        return format_model_answer(raw_result, llm)

    return Tool(
        name="AnswerFormatter",
        func=format_answer,
        description="Formatta e traduce in italiano la risposta con il risultato di una query SQL",
        return_direct=True
    )


def build_custom_agent(llm_key):
    """
    Funzione che inizializza un agente LangChain personalizzato per l'interrogazione di un database SQL
    tramite linguaggio naturale
    - Configura il modello LLM llama3 tramite endpoint Groq, utilizzando l'API key fornita
    - Crea la connessione al database SQLite locale e ottiene il suo schema
    - Costruisce i tool personalizzati per:
        - Validare semanticamente la domanda
        - Generare una query SQL coerente con lo schema
        - Validare la query generata
        - Eseguire la query sul database
        - Formattare e tradurre in italiano il risultato della query
    - Inizializza un agente LangChain con i tool configurati
    :param llm_key: chiave API per autenticare le richieste al provider Groq (OpenAI compatibile)
    :return: agente LangChain configurato con i tool personalizzati
    """
    llm = ChatOpenAI(
        model="llama3-8b-8192",
        temperature=0,
        openai_api_key=llm_key,
        openai_api_base="https://api.groq.com/openai/v1",
    )

    db = SQLDatabase.from_uri("sqlite:///documents.db")
    db_schema = db.get_table_info()

    # Costruisce i tool
    question_validator_tool = build_question_validator_tool(llm, db_schema)
    sql_query_tool = build_sql_query_tool(llm, db)
    query_validator_tool = build_query_validator_tool(llm, db_schema)
    query_executor_tool = build_query_executor_tool(db)
    answer_formatter_tool = build_answer_formatter_tool(llm)

    # Lista dei tool da fornire all'agente
    tools = [
        question_validator_tool,
        sql_query_tool,
        query_validator_tool,
        query_executor_tool,
        answer_formatter_tool
    ]

    # Inizializza l'agente
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        max_iterations=5,
        early_stopping_method="generate"
    )

    return agent, llm, db


def run_agent(llm_key, question):
    """
    Funzione per eseguire un agente LangChain e rispondere a una domanda in linguaggio naturale
    interrogando un database SQL locale
    - Inizializza l'agente personalizzato
    - Recupera lo schema del database per la validazione semantica della domanda
    - Valida la domanda in linguaggio naturale rispetto allo schema del database
    - Se la domanda è valida:
        - Esegue l'agente con input testuale
        - Recupera la query SQL generata e il risultato grezzo ottenuto dal database tramite intermediate_steps
        - Estrae la risposta finale generata
        - Restituisce uno stato "valid_question" con tutte le informazioni raccolte
    - Gestisce i casi di errore o di incompatibilità semantica della domanda
    :param llm_key: chiave API per autenticare le richieste al provider Groq (OpenAI compatibile)
    :param question: stringa contenente la domanda in linguaggio naturale
    :return dizionario contenente lo stato della domanda, la domanda dell'utente, la query SQL generata,
            il risultato grezzo della query eseguita sul database e la risposta finale del modello
    """
    try:
        agent, llm, db = build_custom_agent(llm_key)
        db_schema = db.get_table_info()

        # Valida la domanda rispetto allo schema
        if not is_question_valid_for_db(question, llm, db_schema):
            return {
                "status": "invalid_question",
                "question": question,
                "sql_query": None,
                "raw_result": None,
                "answer": None
            }

        # Esecuzione dell'agente
        response = agent.invoke({"input": question})
        final_answer = response["output"]
        sql_query = None
        raw_result = None

        for action, output in response["intermediate_steps"]:
            if action.tool == "SQLQueryGenerator":
                sql_query = output
            elif action.tool == "QueryExecutor":
                raw_result = output

        return {
            "status": "valid_question",
            "question": question,
            "query": sql_query,
            "raw_result": raw_result,
            "answer": final_answer
        }

    except Exception as e:
        st.write("Error:", str(e))
        return {
            "status": "error",
            "question": question,
            "query": None,
            "raw_result": None,
            "answer": None,
            "error_message": str(e)
        }


def render_llm_interface():
    """
    Funzione per visualizzare l'interfaccia per interrogazioni in linguaggio naturale su database SQL tramite LLM
    - Recupera la chiave API
    - Mostra un messaggio informativo con la descrizione del database per aiutare l'utente a
      formulare domande pertinenti
    - Mostra una selectbox contenente esempi di interrogazioni, che funge anche da campo di input testuale per
      l'inserimento di domande personalizzate
    - Se la domanda è nuova, esegue l'elaborazione NLP→SQL
    - Visualizza lo stato della richiesta, la domanda dell’utente, la query SQL generata, il risultato grezzo del
      database e la risposta finale del modello
    - In caso di domanda non compatibile con lo schema del database, mostra un avviso
    - In caso di errore durante l’elaborazione, mostra il messaggio dell’eccezione sollevata
    """
    llm_key = st.secrets["general"]["GROQ_LLM_KEY"]

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
        res = run_agent(llm_key, user_question)
        st.session_state.llm_result = res
        st.session_state.last_rendered_answer = res["answer"]

    if st.session_state.llm_result:
        res = st.session_state.llm_result

        st.markdown("<h3 style='font-size:18px;'>Query status:</h3>", unsafe_allow_html=True)
        st.write(res['status'])

        match res["status"]:
            case "valid_question":
                st.markdown("<h3 style='font-size:18px;'>Natural language question:</h3>", unsafe_allow_html=True)
                st.write(res["question"])

                st.markdown("<h3 style='font-size:18px;'>Generated SQL query:</h3>", unsafe_allow_html=True)
                st.code(res["query"], language="sql")

                st.markdown("<h3 style='font-size:18px;'>Raw SQL result:</h3>", unsafe_allow_html=True)
                st.code(res["raw_result"], language="python", wrap_lines=True, height="content", width="stretch")

                st.markdown("<h3 style='font-size:18px;'>Model-generated answer:</h3>", unsafe_allow_html=True)
                st.text(res["answer"])

            case "invalid_question":
                st.warning("The question is not compatible with the information in the database. Please"
                           " try asking a different, more suitable question")

            case "error":
                error_msg = res.get("error_message", "An unexpected error occurred")
                st.error(f"An error occurred while answering the question using SQL:\n\n{error_msg}")
