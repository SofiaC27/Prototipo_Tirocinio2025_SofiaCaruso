import os
import base64
# from dotenv import load_dotenv
from groq import Groq


def encode_image(img_path):
    """
    Funzione per codificare l'immagine in Base64
    - Apre il file in lettura binaria
    - Legge il contenuto e lo converte in una stringa in base 64
    - Decodifica in un formato leggibile "utf-8"
    :param img_path: percorso dell'immagine da codificare
    :return: stringa in base 64 dell'immagine
    """
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Recupera la chiave API dall'ambiente
# load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

# Controlla se la chiave API è presente
if not api_key:
    raise ValueError("API Key not found!")


# Recupera il percorso dell'immagine (se è presente) e ne ottiene la stringa in base 64 per passarla all'API
image_name = "scontrino5.jpg"
image_path = os.path.join("..", "Images", image_name)
if not os.path.exists(image_path):
    raise ValueError(f"File not found!: {image_path}")
base64_image = encode_image(image_path)



# Inizializza il client Groq
client = Groq(api_key=api_key)

# Esegue la richiesta di completamento
chat_completion = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Estrai il testo da questa immagine"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ]
)

# Stampa la risposta del modello
print(chat_completion.choices[0].message.content)
