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

# Chiave API
llm_key = st.secrets["general"]["GROQ_LLM_KEY"]


@cl.on_chat_start
async def on_chat_start():
    # Inizializza agente e componenti
    agent, llm, db_schema = build_custom_agent(llm_key)
    cl.user_session.set("agent", agent)
    cl.user_session.set("llm", llm)
    cl.user_session.set("db_schema", db_schema)

    # Descrizione del database
    intro = (
        "Il database contiene informazioni estratte da scontrini: include dati sulle immagini caricate,"
        " dettagli del negozio e della transazione, l’elenco dei prodotti acquistati con eventuali sconti"
        " applicati. È progettato per rispondere a domande relative agli acquisti, ai negozi, ai prezzi, alle date,"
        " ai prodotti e ai metodi di pagamento."
    )
    await cl.Message(content=intro).send()

    # Esempi di domande come pulsanti cliccabili
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

    # Invio degli esempi come azioni interattive (pulsanti)
    actions = []
    for ex in examples:
        action = cl.Action(
            name="esempio_domanda",
            payload={"value": ex},
            label=ex
        )
        actions.append(action)

    await cl.Message(
        content="Ecco alcuni esempi di domande che puoi fare:",
        actions=actions
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    content = message.content.lower().strip()

    if content in GREETING_MESSAGES:
        await cl.Message(content="Ciao! Chiedimi pure, sono qui per aiutarti.").send()
        return

    if content in COURTESY_MESSAGES:
        await cl.Message(content="Prego! Fammi sapere se hai altre domande.").send()
        return

    # Recupera oggetti sessione
    agent = cl.user_session.get("agent")
    llm = cl.user_session.get("llm")
    db_schema = cl.user_session.get("db_schema")

    # Validazione semantica della domanda
    if not is_question_valid_for_db(message.content, llm, db_schema):
        await cl.Message(content="La domanda non è compatibile con il database. Prova a riformularla.").send()
        return

    # Messaggio di attesa
    thinking = cl.Message(content="Sto elaborando la risposta, un attimo di pazienza...")
    await thinking.send()

    try:
        # Esecuzione dell'agente
        response = agent.invoke({"input": message.content})
        final_answer = response["output"]

        sql_query = None
        raw_result = None

        for action, output in response["intermediate_steps"]:
            if action.tool == "SQLQueryGenerator":
                sql_query = output
            elif action.tool == "QueryExecutor":
                raw_result = output

        # Messaggi separati
        if sql_query:
            await cl.Message(content=f"**Query generata:**\n```sql\n{sql_query}\n```").send()

        if raw_result:
            await cl.Message(content=f"**Risultato grezzo:**\n{raw_result}").send()

        # Streaming della risposta finale
        msg = cl.Message(content="**Risposta finale:**\n")
        for token in final_answer.split():
            await msg.stream_token(token + " ")
        await msg.send()

    except Exception as e:
        await cl.Message(content=f"Errore: {str(e)}").send()
