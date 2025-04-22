import pytesseract
from PIL import Image


# NOTE: configura il percorso dell'eseguibile di Tesseract OCR, necessario per il corretto
# funzionamento della libreria pytesseract per il riconoscimento ottico dei caratteri
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def extract_text_from_image(image_path):
    """
    Funzione per estrarre il testo da un'immagine attraverso l'OCR
    - Prende l'immagine corrispondente al percorso indicato
    - Usa la libreria per prendere il testo presente nell'immagine (permette di selezionare la lingua)
    :param image_path: percorso dell'immagine da cui estrarre il testo
    :return: testo estratto dall'immagine
    """
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang="ita")
    return text
