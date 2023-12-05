from .config import TOKENS_TYPE, VECTOR_SIZE, WINDOW, MIN_COUNT, SG
from gensim.models import Word2Vec
import spacy
import os
import numpy as np
from .config import DOCUMENT_EMBEDDING, EMBEDDING_MODEL_TYPE, EMBEDDING_MODEL_FNAME
from pymongo import UpdateOne
import csv
import subprocess


def get_embedding_model():
    if EMBEDDING_MODEL_TYPE == 'Word2Vec':
        embedding_model = Word2Vec.load(os.getcwd(), "question_answer", "embedding_models", EMBEDDING_MODEL_FNAME)
    elif EMBEDDING_MODEL_TYPE.lower() == 'glove':
        # Load the custom spaCy model
        embedding_model = spacy.load(
            os.path.join(os.getcwd(), "question_answer", "embedding_models", EMBEDDING_MODEL_FNAME.split(".bin")[0]))
    return embedding_model


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


def update_embedding_model(df):
    if EMBEDDING_MODEL_TYPE == 'Word2Vec':
        kwargs = {
            'sentences': df[TOKENS_TYPE].to_list(),
            'vector_size': VECTOR_SIZE,
            'window': WINDOW,
            'min_count': MIN_COUNT,
            'sg': SG
        }

        # Train the Word2Vec model
        model = Word2Vec(**kwargs)

        # Save the model
        model.save(os.path.join("..", "models", "word_embeddings", EMBEDDING_MODEL_FNAME))

    elif EMBEDDING_MODEL_TYPE == 'glove':
        # Specify the file path for the output text file
        output_file = os.path.join(os.getcwd(), "question_answer", "embedding_models", "glove",
                                   'training_data.txt')

        # Write the "tokens" column to a text file with each row on a separate line
        df[TOKENS_TYPE].apply(lambda x: ' '.join(x)).to_csv(output_file, header=False, index=False,
                                                                     sep='\n',
                                                                     quoting=csv.QUOTE_NONE)

        os.environ["VECTOR_SIZE"] = str(VECTOR_SIZE)
        os.environ["WINDOW_SIZE"] = str(WINDOW)
        os.environ["VOCAB_MIN_COUNT"] = str(MIN_COUNT)
        # sys.path.append(os.path.join("..", "models", "word_embeddings", "glove"))

        # Train the model
        demo_path = os.path.join(os.getcwd(), "question_answer", "embedding_models", "glove")
        os.chdir(demo_path)
        script_path = os.path.join(demo_path, "demo.sh")
        try:
            # Run the demo.sh script
            subprocess.run([script_path], check=True, shell=True)
            # For example: subprocess.run([script_path, 'arg1', 'arg2'], check=True, shell=True)
        except subprocess.CalledProcessError as e:
            # Handle errors if the subprocess returns a non-zero exit code
            print(f"Error running script: {e}")
        if os.getcwd().endswith('glove'):
            views_path = os.path.join("..", "..", "..")
            os.chdir(os.path.join(views_path))

        # Path to your GloVe vectors file
        vectors_file = os.path.join(os.getcwd(), "question_answer", "embedding_models", "glove", "vectors.txt")

        # Load the custom spaCy model with GloVe vectors
        custom_nlp = load_custom_vectors(vectors_file)

        # Save the custom spaCy model to a directory
        custom_nlp.to_disk(os.path.join(os.getcwd(), "question_answer", "embedding_models",
                                        EMBEDDING_MODEL_FNAME.split(".bin")[0]))

        print("updated the embedding layer")
        return


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
