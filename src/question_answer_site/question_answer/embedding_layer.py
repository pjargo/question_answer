from spacy.vocab import Vocab
from spacy.language import Language
from spacy.tokens import Doc
from gensim.models import Word2Vec
import spacy
import os
import numpy as np
from .config import DOCUMENT_EMBEDDING, EMBEDDING_MODEL_TYPE, EMBEDDING_MODEL_FNAME
from pymongo import UpdateOne


def get_embedding_model():
    if EMBEDDING_MODEL_TYPE == 'Word2Vec':
        model = Word2Vec.load(os.getcwd(), "question_answer", "embedding_models", EMBEDDING_MODEL_FNAME)
    elif EMBEDDING_MODEL_TYPE.lower() == 'glove':
        # Load the custom spaCy model
        model = spacy.load(
            os.path.join(os.getcwd(), "question_answer", "embedding_models", EMBEDDING_MODEL_FNAME.split(".bin")[0]))
    return model


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
        print(counter_value)
        mongodb.update_document(query, update)


def update_mongo_documents_bulk(rows, mongodb):
    bulk_operations = []

    for index, row in rows.iterrows():
        counter_value = row['counter']
        tokens_value = row[DOCUMENT_EMBEDDING]

        # Assuming your MongoDB documents have a unique identifier field 'counter'
        query = {'counter': counter_value}
        update = {'$set': {DOCUMENT_EMBEDDING: tokens_value}}

        bulk_operations.append(UpdateOne(query, update))

    try:
        # Execute the bulk update
        if mongodb.connect():
            result = mongodb.bulk_update_documents(bulk_operations)
            print(f"Updated {result.modified_count} documents")
    except Exception as e:
        print(f"Error during bulk update: {e}")
    finally:
        # Make sure to close the connection
        mongodb.disconnect()
