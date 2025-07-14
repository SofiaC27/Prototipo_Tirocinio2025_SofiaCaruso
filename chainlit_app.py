import chainlit as cl
import streamlit as st

from Modules.llm_functions import is_question_valid_for_db, build_custom_agent

# Frasi da filtrare
COURTESY_MESSAGES = [
    "grazie", "grazie mille", "ti ringrazio", "ok", "ok grazie", "va bene",
    "va bene grazie", "capito", "tutto chiaro", "ricevuto", "bene",
    "perfetto", "chiaro", "ottimo", "eccellente"
]

GREETING_MESSAGES = [
    "ciao", "salve", "buongiorno", "buonasera", "hey", "ehi"
]

llm_key = st.secrets["general"]["GROQ_LLM_KEY"]


@cl.on_chat_start
async def on_chat_start():
    agent, llm, db_schema = build_custom_agent(llm_key)
    cl.user_session.set("agent", agent)
    cl.user_session.set("llm", llm)
    cl.user_session.set("db_schema", db_schema)


@cl.on_message
async def on_message(message: cl.Message):
    content = message.content.lower().strip()

    if content in GREETING_MESSAGES:
        await cl.Message(content="Ciao! Chiedimi pure, sono qui per aiutarti.").send()
        return

    if content in COURTESY_MESSAGES:
        await cl.Message(content="Prego! Fammi sapere se hai altre domande.").send()
        return

    agent = cl.user_session.get("agent")
    llm = cl.user_session.get("llm")
    db_schema = cl.user_session.get("db_schema")

    if not is_question_valid_for_db(message.content, llm, db_schema):
        await cl.Message(content="La domanda non è compatibile con il database. Prova a riformularla.").send()
        return

    try:
        response = agent.invoke({"input": message.content})
        final_answer = response["output"]

        sql_query = None
        raw_result = None

        for action, output in response["intermediate_steps"]:
            if action.tool == "SQLQueryGenerator":
                sql_query = output
            elif action.tool == "QueryExecutor":
                raw_result = output

        reply = (f"**Domanda**: {message.content}\n\n **Query generata**:\n```sql\n{sql_query}\n```\n"
                 f" **Risultato**:\n{raw_result}\n\n **Risposta finale**:\n{final_answer}")
        await cl.Message(content=reply).send()

    except Exception as e:
        await cl.Message(content=f"Errore: {str(e)}").send()
