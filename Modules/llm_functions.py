import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool

from Modules.ocr_groq import load_prompt


def init_chain(api_key):
    """
    Funzione per inizializzare la catena LangChain per interrogazioni in linguaggio naturale su database SQL
    - Configura il modello LLM llama3 tramite endpoint Groq, utilizzando l'API key fornita
    - Crea la connessione al database locale SQLite
    - Costruisce un SQLDatabaseToolkit che l'LLM userà per analizzare ed eseguire query
    - Genera un agente LangChain che traduce domande in linguaggio naturale in query SQL valide, le esegue
      sul database e interpreta i risultati
    - Abilita l'output dettagliato nel terminale (verbose)
    :param api_key: chiave API per autenticare le richieste al provider Groq (OpenAI compatibile)
    :return: oggetto AgentExecutor configurato per gestire query NL → SQL → risultati
    :return: modello LLM configurato per generare query e risposte
    """
    llm = ChatOpenAI(
        model="llama3-8b-8192",
        temperature=0,
        openai_api_key=api_key,
        openai_api_base="https://api.groq.com/openai/v1",
    )

    db = SQLDatabase.from_uri("sqlite:///documents.db")

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_executor_kwargs={"handle_parsing_errors": True}
    )

    return agent_executor, llm


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

    return result.strip().lower() == "true"


def run_nl_query(question, agent_executor, llm):
    """
    Funzione per elaborare una domanda in linguaggio naturale ed eseguire una query SQL tramite un agente LangChain
    - Estrae l'oggetto database dai tool assegnati all'agente
    - Recupera lo schema SQL del database corrente
    - Verifica se la domanda è semanticamente compatibile con lo schema del database tramite un LLM esterno
    - Se la domanda è incompatibile, restituisce uno stato "invalid_question" e risposta nulla
    - Se la domanda è compatibile: esegue l'agente con input testuale, estrae la risposta finale generata dal modello
      e restituisce uno stato "valid_question" e la risposta estratta
    - In caso di errore durante il processo, restituisce uno stato "error" con il messaggio dell'eccezione
    :param question: stringa contenente la domanda in linguaggio naturale dell'utente
    :param agent_executor: istanza AgentExecutor configurata per gestire interrogazioni NL→SQL
    :param llm: istanza LLM compatibile con LangChain, usata per validazione e generazione
    :return: dizionario con la domanda dell'utente, lo stato della domanda e la risposta
    """
    try:
        db_obj = None

        for tool in agent_executor.tools:
            if isinstance(tool, QuerySQLDatabaseTool):
                db_obj = tool.db
                break

        if db_obj is None:
            raise ValueError("SQL tool not found")

        db_schema = db_obj.get_table_info()

        # Verifica se la domanda ha senso rispetto allo schema del DB
        if not is_question_valid_for_db(question, db_schema, llm):
            return {
                "question": question,
                "answer": None,
                "status": "invalid_question"
            }

        # Esegue l'agente con la domanda
        response = agent_executor.invoke({"input": question})
        final_answer = response.get("output", "")

        return {
            "question": question,
            "answer": final_answer,
            "status": "valid_question"
        }

    except Exception as e:
        return {
            "question": question,
            "answer": None,
            "status": "error",
            "error_message": str(e)
        }


def render_llm_interface():
    """
    Funzione per visualizzare l'interfaccia per interrogazioni in linguaggio naturale su database SQL tramite LLM
    - Recupera la chiave API e inizializza l'agente SQL e il modello LLM
    - Mostra un messaggio di info con la descrizione del database per aiutare l'utente a fare domande pertinenti
    - Mostra una selectbox con degli esempi di domande, che fa anche da casella di input testuale per inserire
      una domanda in linguaggio naturale
    - Esegue la funzione di query NLP→SQL usando l'agente e il modello LLM
    - Visualizza i risultati strutturati: stato, domanda e risposta testuale generata dal modello
    - In caso di domanda non compatibile con lo schema del database, mostra un messaggio di avviso
    - In caso di errore durante l'elaborazione, mostra il messaggio dell'eccezione sollevata
    """
    llm_key = st.secrets["general"]["GROQ_LLM_KEY"]

    if "llm_agent" not in st.session_state or "llm" not in st.session_state:
        agent_executor, llm = init_chain(llm_key)
        st.session_state.llm_agent = agent_executor
        st.session_state.llm = llm

    if "llm_result" not in st.session_state:
        st.session_state.llm_result = None

    st.info("The database stores information extracted from receipts: it includes data about"
            " uploaded receipt images, store and transaction details, and the list of purchased items"
            " with potential discounts. It is designed to answer questions about purchases, stores, "
            " prices, dates, products, and payment methods.")

    examples = [
        "Mostra tutti gli scontrini del 2025",
        "Elenca i prodotti acquistati con lo sconto",
        "Quanto ho speso a marzo?",
        "Mostra i prodotti acquistati più di una volta",
        "Qual è stato il mese con la spesa più alta?"
    ]

    user_question = st.selectbox(
        "Enter a question or choose one from the examples below:",
        options=examples,
        index=None,
        placeholder="Type or select a question...",
        accept_new_options=True,
        key="nl_input"
    )

    if user_question:
        res = run_nl_query(user_question, st.session_state.llm_agent, st.session_state.llm)
        st.session_state.llm_result = res

    if "llm_result" in st.session_state and st.session_state.llm_result:
        res = st.session_state.llm_result

        st.markdown("# Query status:")
        st.write(res['status'])

        match res["status"]:
            case "valid_question":
                st.markdown("# Natural language question:")
                st.write(res["question"])

                st.markdown("# Model-generated answer:")
                st.text(res["answer"])

            case "invalid_question":
                st.warning("The question is not compatible with the information in the database. Please"
                           " try asking a different, more suitable question")

            case "error":
                error_msg = res.get("error_message", "An unexpected error occurred")
                st.error(f"An error occurred while answering the question using SQL:\n\n{error_msg}")
