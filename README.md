# Smart Receipts

Un'applicazione web per caricare scontrini, estrarre dati tramite OCR e salvarli in un database ricercabile.
L'app è potenziata con AI/LLM per interrogazioni in linguaggio naturale e con ML per analisi avanzate.

## Funzionalità principali
- Upload di immagini di scontrini
- OCR automatico con estrazione di dati strutturati in formato JSON
- Salvataggio dei dati su database ricercabile
- Interrogazioni e ricerca tramite modelli AI/LLM in linguaggio naturale
- Analisi con ML per rilevamento e classificazione automatica di outlier nei dati

## Requisiti e dipendenze
Tutte le librerie necessarie sono elencate nel file `requirements.txt`
Per installarle, esegui il comando:
pip install -r requirements.txt

## Esecuzione
Per avviare l'applicazione, è necessario eseguire due comandi in parallelo, uno per Streamlit e uno per Chainlit.
Apri due terminali separati e lancia:
streamlit run app.py
chainlit run chainlit_app.py -w --headless
