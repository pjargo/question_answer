import spacy
from spacy.vocab import Vocab
from spacy.language import Language
from spacy.tokens import Doc

import numpy as np

from .config import DOCUMENT_EMBEDDING


# Load your GloVe vectors into a custom spaCy model
def load_custom_vectors(vectors_path):
    nlp = spacy.blank("en")
    with open(vectors_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(" ")
            word = parts[0]
            vector = [float(val) for val in parts[1:]]
            vector = np.array(vector)
            nlp.vocab.set_vector(word, vector)
    return nlp


# Function to update MongoDB documents based on DataFrame values
def update_mongo_document(row, mongodb):
    counter_value = row['counter']
    tokens_value = row[DOCUMENT_EMBEDDING]
    # Assuming your MongoDB documents have a unique identifier field 'counter'
    query = {'counter': counter_value}
    update = {'$set': {DOCUMENT_EMBEDDING: tokens_value}}

    # Update the MongoDB document
    if mongodb.connect():
        mongodb.update_document(query, update)
