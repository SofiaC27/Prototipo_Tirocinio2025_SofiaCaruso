import os

# Base directory: la root del progetto, dove si trova config.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Percorsi assoluti delle cartelle
IMAGE_DIR_PATH = os.path.join(BASE_DIR, "Images")
EXTRACTED_JSON_DIR_PATH = os.path.join(BASE_DIR, "Extracted_JSON")

# Solo i nomi delle cartelle
IMAGE_DIR = os.path.basename(IMAGE_DIR_PATH)
EXTRACTED_JSON_DIR = os.path.basename(EXTRACTED_JSON_DIR_PATH)
