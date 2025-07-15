import chainlit as cl
import streamlit as st
import ast

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

MAX_RIGHE = 30  # numero massimo di righe consentite

# Chiave API
llm_key = st.secrets["general"]["GROQ_LLM_KEY"]


@cl.action_callback("esempio_domanda")
async def question_action_handler(action: cl.Action):
    """
    Funzione che gestisce il clic su un pulsante (esempio_domanda) nella chat Chainlit
    - Recupera il contenuto dal pulsante cliccato
    - Simula l'invio della domanda come se fosse stata scritta dall'utente
    :param action: oggetto cl.Action contenente la domanda
    """
    domanda = action.payload["value"]
    await on_message(cl.Message(content=domanda))


@cl.on_chat_start
async def on_chat_start():
    """
    Funzione di avvio della chat Chainlit
    - Inizializza l’agente LangChain e memorizza il modello, l'agente e lo schema del database
    - Mostra un messaggio introduttivo con descrizione del database
    - Invia una serie di esempi interattivi di domande come pulsanti con icone e tooltip
    """
    # Inizializza agente e componenti
    agent, llm, db_schema = build_custom_agent(llm_key)
    cl.user_session.set("agent", agent)
    cl.user_session.set("llm", llm)
    cl.user_session.set("db_schema", db_schema)

    # Introduzione all'assistente e descrizione del database
    intro = (
        "👋 Ciao! Sono il tuo assistente intelligente, qui per aiutarti a esplorare e interrogare il database"
        " degli acquisti basati sugli scontrini che sono stati caricati. Posso rispondere alle tue domande,"
        " filtrare informazioni rilevanti e generare riepiloghi personalizzati sulle spese e sulle abitudini di"
        " consumo.\n\n 📄 Il database contiene informazioni estratte dalle immagini degli scontrini, come i dettagli"
        " dei negozi, le transazioni, i prodotti acquistati ed eventuali sconti applicati. È pensato per offrirti"
        " risposte su tutto ciò che riguarda acquisti, prezzi, negozi, date, metodi di pagamento e frequenza"
        " di spesa.\n\n 🗨️ Scrivimi una domanda oppure seleziona uno degli esempi qui sotto: sono pronto ad"
        " aiutarti! 😊"
    )
    await cl.Message(content=intro).send()

    # Esempi di domande come pulsanti cliccabili e icone
    examples = {
        "Mostrami i primi 15 scontrini caricati nel 2025": "receipt-euro",
        "Mostrami i primi 10 acquisti effettuati nel 2025": "shopping-cart",
        "Elenca i prodotti per cui è stato applicato uno sconto": "percent",
        "Qual è la somma totale delle spese effettuate nel mese di marzo?": "calendar-days",
        "Quali prodotti sono stati acquistati più di una volta in giorni diversi?": "repeat",
        "In quale mese del 2025 ho speso di più in totale?": "calendar-clock",
        "Quali negozi ho visitato più spesso?": "map-pin",
        "Qual è stato il metodo di pagamento più usato nei miei acquisti?": "credit-card",
        "Mostrami tutti i prodotti acquistati in contanti": "wallet",
        "Quali sono i prodotti più acquistati in termini di quantità totale?": "chart-bar"
    }

    # Invio degli esempi come azioni interattive (pulsanti)
    actions = []
    for question, icon in examples.items():
        action = cl.Action(
            name="esempio_domanda",
            payload={"value": question},
            label=question,
            icon=icon,
            tooltip="Domanda di esempio"
        )
        actions.append(action)

    await cl.Message(
        content="Ecco alcuni esempi di domande che puoi fare:",
        actions=actions
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """
    Funzione che gestisce ogni nuovo messaggio dell’utente
    - Filtra messaggi di cortesia o saluto per risposte rapide
    - Valida la domanda rispetto allo schema del database
    - Invoca l’agente LangChain e recupera la query, il risultato SQL e la risposta finale
    - Mostra messaggi distinti per query, risultato grezzo e risposta finale
    - Se il risultato ha esattamente MAX_RIGHE righe, mostra un avviso di limitazione
    :param message: oggetto cl.Message contenente il testo dell’utente
    """
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
                if isinstance(raw_result, str):
                    raw_result = ast.literal_eval(raw_result)

        # Avviso se il risultato supera il limite
        if raw_result and isinstance(raw_result, list) and len(raw_result) == MAX_RIGHE:
            await cl.Message(
                content=f"⚠️ La risposta è stata limitata ai primi {MAX_RIGHE} elementi per garantire una maggiore"
                        f" velocità e stabilità"
            ).send()

        # Messaggi separati
        await cl.Message(content=f"**Domanda:**\n{message.content}").send()

        if sql_query:
            await cl.Message(content=f"**Query generata:**\n```sql\n{sql_query}\n```").send()

        if raw_result:
            await cl.Message(content=f"**Risultato grezzo:**\n{raw_result}").send()

        await cl.Message(content=f"**Risposta finale:**\n{final_answer}").send()

    except Exception as e:
        await cl.Message(content=f"Errore: {str(e)}").send()
